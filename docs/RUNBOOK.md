# Runbook de Implantação e Operação em Produção — xau-bot

Este documento é o guia definitivo para operadores, mantenedores e engenheiros de DevOps implantarem, configurarem e operarem o robô institucional SMC de XAU/USD (`xau-bot`) em ambientes de produção baseados em VPS Windows Server (2022/2019) ou Windows 10/11 x64.

---

## 1. Visão Geral da Arquitetura em Produção

O robô opera através de uma arquitetura hexagonal com segregação estrita de responsabilidades e mecanismo de defesa em camadas duplas:

```
               ┌─────────────────────── VPS Windows (Datacenter LD4 / NY4) ───────────────────────┐
               │                                                                                   │
Corretora MT5  │  ┌────────────────────────┐                   ┌───────────────────────────────┐  │
(Servidor Live)│  │   terminal64.exe       │                   │    xau-bot (Python 3.12)      │  │
      ▲        │  │   (MetaTrader 5)       │                   │    (Serviço Windows - NSSM)   │  │
      │        │  │                        │                   │                               │  │
      │ IPC/   │  │  ┌──────────────────┐  │  Common\Files     │  ┌─────────────────────────┐  │  │
      │ TCP    │  │  │ XauGuardian.mq5  │  │ (Heartbeat / KS / │  │ Watchdog / Candle Loop │  │  │
      └────────┼─►│  │ (EA no Gráfico   │◄─┼─ Calendário JSON) ┼─►│ Execution / RiskGuard   │  │  │
               │  │  │  M1 XAUUSD)      │  │                   │  │ Filters / SMC Strategy  │  │  │
               │  │  └──────────────────┘  │                   │  └───────────┬─────────────┘  │  │
               │  └───────────▲────────────┘                   └──────────────┼────────────────┘  │
               │              │ IPC Local                                     │                   │
               │              └───────────────────────────────────────────────┤                   │
               │                                                              ▼                   │
               │                                                HTTP 127.0.0.1:8000/health        │
               │                                                (Liveness & Observabilidade)      │
               │                                                              │                   │
               └──────────────────────────────────────────────────────────────┼───────────────────┘
                                                                              ▼
                                                                  Alertas Telegram / Uptime Kuma
```

### Princípios Chave de Confiabilidade
1. **Defesa em Duas Camadas:**
   - **Camada 1 (Servidor da Corretora):** Todas as ordens a mercado ou limites são enviadas com Stop Loss e Take Profit associados diretamente na corretora. Caso a VPS sofra pane física instantânea, o capital está protegido por stops residentes no servidor.
   - **Camada 2 (EA Guardião MQL5):** Se o processo Python travar, congelar em loop infinito ou silenciar por mais de 30 segundos, o EA nativo `XauGuardian.mq5` fecha posições e cancela ordens pendentes. Além disso, se a perda diária exceder 3% ou o drawdown da conta exceder 10%, o EA aciona o kill switch na corretora.
2. **Idempotência e Reconciliação:** O terminal e o servidor da corretora são a fonte da verdade. O bot reconcilia o estado da conta a cada início de sessão ou reconexão.
3. **Segurança de Segredos:** Credenciais e tokens residem exclusivamente nas variáveis de ambiente do sistema operacional, nunca em logs ou arquivos versionados.

---

## 2. Requisitos e Dimensionamento de VPS

A escolha e localização física do servidor virtual são cruciais para a execução determinística da estratégia SMC em ouro:

| Parâmetro | Requisito Mínimo | Recomendado em Produção | Justificativa |
| :--- | :--- | :--- | :--- |
| **Localização** | Mesma região da corretora | Londres (LD4/Slough) ou NY (NY4) | Latência de ping com o servidor MT5 < 5ms. |
| **Sistema Operacional** | Windows 10 x64 | Windows Server 2022 Standard x64 | Estabilidade em long-running services e controle fino de GPO. |
| **Processador** | 2 vCPUs (2.4 GHz+) | 4 vCPUs dedicadas (3.0 GHz+) | Cálculo rápido de POIs SMC no fechamento de vela sem latência. |
| **Memória RAM** | 4 GB | 8 GB | Suporta terminal MT5 + Python + buffers de histórico em memória. |
| **Armazenamento** | 40 GB SSD | 80 GB NVMe SSD | I/O rápido para ticks, histórico e rotação de logs. |
| **Conectividade** | 100 Mbps com IP fixo | 1 Gbps redundante com SLA 99.9% | Evita quedas espúrias de IPC e reconexões frequentes. |

---

## 3. Provisionamento do Sistema Operacional

Execute os passos a seguir na VPS recém-criada através de uma sessão com privilégios de Administrador:

### 3.1. Instalação do Python 3.12 x64
1. Baixe o instalador oficial do Python 3.12 (ex: Python 3.12.10 x64) em `https://www.python.org/downloads/windows/`.
2. Execute o instalador selecionando:
   - `[x] Add python.exe to PATH`
   - `[x] Use admin privileges when installing py.exe`
   - Clique em **Customize installation** e marque `[x] Install for all users`.
3. Verifique a instalação no PowerShell:
   ```powershell
   python --version
   # Deve retornar: Python 3.12.x
   ```

### 3.2. Instalação do PowerShell 7 (`pwsh`)
O PowerShell 7 oferece maior desempenho e suporte moderno a UTF-8:
```powershell
winget install --id Microsoft.PowerShell --source winget
```

### 3.3. Instalação do NSSM (Non-Sucking Service Manager)
O NSSM gerencia o processo Python como serviço do Windows com recuperação e rotação de logs:
1. Baixe o release estável do NSSM (`nssm-2.24.zip`) de `https://nssm.cc/download`.
2. Descompacte e copie `win64\nssm.exe` para `C:\tools\nssm\nssm.exe`.
3. Adicione `C:\tools\nssm` ao PATH do sistema:
   ```powershell
   $currentPath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
   if ($currentPath -notlike "*C:\tools\nssm*") {
       [Environment]::SetEnvironmentVariable("PATH", "$currentPath;C:\tools\nssm", "Machine")
   }
   ```

---

## 4. Sincronização de Relógio de Alta Precisão (NTP)

No trading institucional, desvios de relógio geram descasamento de velas de M5/M1 e falhas no filtro de notícias. O serviço `w32time` deve sincronizar frequentemente com servidores NTP de alta precisão.

### 4.1. Configuração do Serviço Windows Time
Execute no PowerShell como Administrador:
```powershell
# 1. Parar o serviço temporariamente
net stop w32time

# 2. Configurar servidores NTP e intervalo de polling rápido (900s = 15 minutos)
w32tm /config /syncfromflags:manual /manualpeerlist:"pool.ntp.org,0x1 time.windows.com,0x1" /reliable:yes /update
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Services\W32Time\TimeProviders\NtpClient" -Name "SpecialPollInterval" -Value 900 -Type DWord

# 3. Iniciar e registrar serviço para auto-inicialização
net start w32time
Set-Service -Name w32time -StartupType Automatic

# 4. Forçar ressincronização imediata
w32tm /resync /rediscover
```

### 4.2. Auditoria de Offset e Drift
Para verificar se o relógio está perfeitamente sincronizado:
```powershell
w32tm /query /status
```
> [!TIP]
> O campo **Source** deve indicar um dos servidores configurados (ex: `pool.ntp.org`) e o **Precision** deve estar em torno de `-23` (~100 nanossegundos).

---

## 5. Prevenção de Reinicializações do Windows Update

> [!CAUTION]
> **Risco Crítico:** Uma reinicialização não programada do Windows Update durante a sessão de Londres ou Nova York com posições abertas ou ordens pendentes pode gerar perdas financeiras graves.

### 5.1. Desativação de Auto-Reboot via Registro
Execute no PowerShell como Administrador para bloquear reinícios automáticos enquanto houver usuários logados:
```powershell
# Cria a chave de política se não existir
$keyPath = "HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU"
if (-not (Test-Path $keyPath)) {
    New-Item -Path $keyPath -Force | Out-Null
}

# 1. Impede reinício automático com usuários logados
Set-ItemProperty -Path $keyPath -Name "NoAutoRebootWithLoggedOnUsers" -Value 1 -Type DWord

# 2. Configura notificações sem download/instalação silenciosa espontânea (AUOptions = 2 ou 3)
Set-ItemProperty -Path $keyPath -Name "AUOptions" -Value 2 -Type DWord

# Forçar atualização de políticas locais
gpupdate /force
```

### 5.2. Janela Oficial de Manutenção
- **Janela Permitida:** Sábados entre 10:00 UTC e 18:00 UTC (mercado forex 100% fechado em todo o mundo).
- **Procedimento:**
  1. Parar o serviço: `pwsh scripts\service-nssm.ps1 -Action Stop`.
  2. Fechar o terminal MT5.
  3. Aplicar atualizações pendentes do Windows e reiniciar o servidor manualmente.
  4. Realizar o checklist de subida antes da abertura do mercado no domingo às 18:00 EST.

---

## 6. Instalação e Configuração do MetaTrader 5

### 6.1. Configuração do Terminal
1. Instale o MetaTrader 5 fornecido pela corretora no diretório padrão (`C:\Program Files\MetaTrader 5`).
2. Abra o terminal e conecte à conta de negociação:
   - Marque obrigatoriamente a opção **"Salvar Senha"**.
3. No menu **Ferramentas $\to$ Opções $\to$ Expert Advisors**:
   - `[x] Permitir Algo Trading`
   - Desmarque: *"Desativar Algo Trading ao trocar de conta"* (se aplicável ao perfil operacional).
   - Desmarque: *"Desativar Algo Trading ao trocar de perfil"*.

### 6.2. Instalação do EA Guardião MQL5
1. Copie o arquivo [`xau-bot/guardian/XauGuardian.mq5`](../guardian/XauGuardian.mq5) para a pasta de Experts do MT5:
   `%APPDATA%\MetaQuotes\Terminal\<INSTANCE_ID>\MQL5\Experts\XauGuardian.mq5`
2. Abra o **MetaEditor** (pressione `F4` no MT5), abra `XauGuardian.mq5` e pressione **F7** para compilar (gerando `XauGuardian.ex5` com 0 erros).
3. No MT5, abra um gráfico dedicado de **XAUUSD M1** e anexe o EA `XauGuardian`:
   - Na aba **Comum**, marque `[x] Permitir Algo Trading`.
   - Na aba **Inputs**, confirme os parâmetros:
     - `InpMagicNumber`: `202610`
     - `InpHeartbeatTimeoutSec`: `30`
     - `InpMaxDailyLossPct`: `3.0`
     - `InpMaxDrawdownPct`: `10.0`
     - `InpCalendarExportIntervalMin`: `60`
4. Verifique o ícone do chapéu no canto superior direito do gráfico: deve estar **azul** (Algo Trading ativo).

### 6.3. Inicialização Automática e Auto-Logon do Windows
O MetaTrader 5 é uma aplicação desktop gráfica com interface de usuário. Para que o terminal inicialize automaticamente após reinicializações da VPS (ex: manutenções de fim de semana), é necessário configurar o logon automático da conta do Windows ou uma Tarefa Agendada no boot:

#### Opção A: Auto-Logon Seguro via Microsoft Sysinternals Autologon (Recomendado)
O utilitário oficial Sysinternals Autologon cifra as credenciais do usuário na LSA (Local Security Authority), evitando expor senhas em texto puro no registro do Windows:
1. Baixe o Autologon oficial da Microsoft: `https://learn.microsoft.com/en-us/sysinternals/downloads/autologon`.
2. Execute no PowerShell como Administrador:
   ```cmd
   autologon.exe Administrator . "SuaSenhaSeguraAqui"
   ```
3. Crie o atalho do MT5 na pasta Startup do usuário:
   ```powershell
   $startupFolder = [System.Environment]::GetFolderPath('Startup')
   $shortcutPath = Join-Path $startupFolder "MetaTrader 5.lnk"
   $wsh = New-Object -ComObject WScript.Shell
   $sc = $wsh.CreateShortcut($shortcutPath)
   $sc.TargetPath = "C:\Program Files\MetaTrader 5\terminal64.exe"
   $sc.Save()
   ```

#### Opção B: Tarefa Agendada no Boot (Sem Autologon de Desktop)
Caso a política corporativa proíba auto-logon interativo, crie uma tarefa no Windows Task Scheduler disparada "Ao iniciar o computador" executando com a conta do operador do MT5:
```powershell
$action = New-ScheduledTaskAction -Execute "C:\Program Files\MetaTrader 5\terminal64.exe"
$trigger = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U -RunLevel Highest
Register-ScheduledTask -TaskName "AutoStart_MT5" -Action $action -Trigger $trigger -Principal $principal
```

> [!WARNING]
> **Limitações da Inicialização sem Sessão Interativa (Opção B):**
> O MetaTrader 5 é concebido como uma aplicação Win32 desktop que espera um subsistema gráfico. Se iniciado em sessão de segundo plano sem console/área de trabalho (modo não-interativo puro), determinadas APIs de gráficos, exportação de arquivos para `Common\Files` e sub-processos do terminal podem ser suspensos pelo Windows. Por isso, a **Opção A (Auto-Logon via Sysinternals Autologon)** é fortemente recomendada para ambientes de produção. Caso opte pela Opção B, configure o MT5 no modo `/portable` e valide previamente a gravação contínua dos arquivos pelo EA Guardião.

---

## 7. Instalação e Deploy do Código `xau-bot`

1. Clone ou transfira o repositório para o diretório de produção:
   `C:\Trading\xau-bot`
2. No diretório `C:\Trading\xau-bot`, crie o ambiente virtual e instale o pacote:
   ```powershell
   cd C:\Trading\xau-bot
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -e .
   ```
3. Execute o script de validação pré-voo (após realizar a configuração de variáveis de ambiente na Seção 8):
   ```powershell
   pwsh scripts\verify-vps.ps1 -Detailed
   ```
   > Todos os itens críticos devem retornar `[PASS]`.


---

## 8. Configuração de Variáveis de Ambiente e Segredos

As configurações do robô são injetadas estritamente via variáveis de ambiente com prefixo `XAU_`.

### 8.1. Tabela Completa de Variáveis Suportadas

| Variável | Tipo | Default | Descrição |
| :--- | :--- | :--- | :--- |
| `XAU_MT5_SERVER` | String | `None` | Nome exato do servidor da corretora no MT5 (ex: `ICMarketsSC-Demo`). |
| `XAU_MT5_LOGIN` | SecretStr | `None` | Número da conta MT5. |
| `XAU_MT5_PASSWORD` | SecretStr | `None` | Senha da conta MT5. |
| `XAU_MT5_PATH` | String | `None` | Caminho explícito para `terminal64.exe` (opcional). |
| `XAU_MT5_TIMEOUT_MS` | Int | `60000` | Timeout em ms para chamadas MT5 IPC (default: 60s). |
| `XAU_BROKER_SERVER_TZ` | String | `UTC` | Timezone do servidor da corretora (ex: `UTC`, `EET`, `Europe/Athens`). Crítico para filtros de sessão SMC. |
| `XAU_GUARDIAN_COMMON_DIR` | String | `None` | **Obrigatório em produção:** Caminho absoluto para a pasta `Common\Files` do MT5 (ex: `C:\Users\Trader\AppData\Roaming\MetaQuotes\Terminal\Common\Files`). |
| `XAU_GUARDIAN_CALENDAR_MAX_AGE_H` | Float | `24.0` | Idade máxima do `calendar.json` antes de rejeitar com `NEWS_CALENDAR_STALE`. |
| `XAU_GUARDIAN_HEARTBEAT_INTERVAL_S`| Float | `5.0` | Intervalo em segundos para escrita do heartbeat no disco. |
| `XAU_GUARDIAN_HEARTBEAT_TIMEOUT_S` | Float | `30.0` | Limiar em segundos para detecção de silêncio pelo guardião. |
| `XAU_GUARDIAN_MAGIC` | Int | `202610` | Magic Number do robô institucional. |
| `XAU_TELEGRAM_TOKEN` | SecretStr | `None` | Token do bot de alertas no Telegram. |
| `XAU_TELEGRAM_CHAT_ID` | SecretStr | `None` | ID do chat/canal Telegram para entrega de alertas operacionais. |
| `XAU_SERVER_HOST` | String | `127.0.0.1` | Interface de escuta do servidor HTTP local. |
| `XAU_SERVER_PORT` | Int | `8000` | Porta local do servidor HTTP de health e status. |
| `XAU_ENV` | String | `development` | Ambiente de execução (`production`, `staging`, `development`). |
| `XAU_MAX_DAILY_LOSS_PCT` | Float | `3.0` | Limite de perda diária no Python (3%). |
| `XAU_MAX_DRAWDOWN_PCT` | Float | `10.0` | Limite de drawdown total no Python (10%). |
| `XAU_MAX_CONSECUTIVE_LOSSES` | Int | `3` | Limite de perdas consecutivas antes de pausar a sessão. |
| `XAU_MAX_DAILY_TRADES` | Int | `5` | Número máximo de trades permitidos no mesmo dia. |
| `XAU_RISK_PER_TRADE_PCT` | Float | `1.0` | Risco percentual por trade (1%). |

### 8.2. Script de Injeção Segura no Escopo da Máquina
Execute o bloco abaixo substituindo pelos dados reais da conta na VPS:
```powershell
[System.Environment]::SetEnvironmentVariable("XAU_MT5_SERVER", "NomeDoServidor-Demo", "Machine")
[System.Environment]::SetEnvironmentVariable("XAU_MT5_LOGIN", "12345678", "Machine")
[System.Environment]::SetEnvironmentVariable("XAU_MT5_PASSWORD", "SuaSenhaSeguraAqui", "Machine")
[System.Environment]::SetEnvironmentVariable("XAU_GUARDIAN_COMMON_DIR", "C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files", "Machine")
[System.Environment]::SetEnvironmentVariable("XAU_TELEGRAM_TOKEN", "123456:ABC-DEF...", "Machine")
[System.Environment]::SetEnvironmentVariable("XAU_TELEGRAM_CHAT_ID", "-100123456789", "Machine")
[System.Environment]::SetEnvironmentVariable("XAU_BROKER_SERVER_TZ", "UTC", "Machine")
[System.Environment]::SetEnvironmentVariable("XAU_ENV", "production", "Machine")
```

> [!NOTE]
> **Atualização da Sessão Atual do PowerShell:**
> `SetEnvironmentVariable(..., "Machine")` grava as variáveis no registro do Windows, mas **não altera as variáveis do processo PowerShell já aberto**. Para que o pré-voo (`verify-vps.ps1`) enxergue as novas variáveis, feche e abra uma **nova janela do PowerShell como Administrador**, ou defina-as também na sessão corrente com `$env:XAU_MT5_SERVER = ...`.

> [!IMPORTANT]
> **Herança de Variáveis no Serviço NSSM:**
> Variáveis de ambiente configuradas no escopo `Machine` só são herdadas pelo subsistema de serviços do Windows (`services.exe`) após uma **reinicialização da VPS (reboot)**.
> Caso precise atualizar variáveis sem reiniciar a máquina inteira, injete-as diretamente nas propriedades extras do serviço via NSSM antes de reiniciá-lo.
> **Atenção:** cada invocação de `AppEnvironmentExtra` substitui a lista anterior por completo. Para configurar múltiplas variáveis, passe todos os pares chave=valor como argumentos separados na mesma chamada:
> ```powershell
> nssm set XauBot AppEnvironmentExtra "XAU_GUARDIAN_COMMON_DIR=C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\Common\Files" "XAU_BROKER_SERVER_TZ=UTC" "XAU_ENV=production"
> pwsh scripts\service-nssm.ps1 -Action Restart
> ```


---

## 9. Gerenciamento do Serviço Windows (NSSM)

Utilize o script de automação [`scripts\service-nssm.ps1`](../scripts/service-nssm.ps1):

### 9.1. Comandos de Operação

#### Passo 1: Instalar o Serviço
Instale o serviço com adaptador MT5 na porta 8000 (executando CLI run):
```powershell
pwsh scripts\service-nssm.ps1 -Action Install -Broker mt5 -Port 8000
```

#### Passo 2: Configurar Conta de Execução do Serviço (Permissões de MT5 IPC)
Por padrão, o NSSM registra serviços executando como a conta de sistema `LocalSystem`. No entanto, como o MetaTrader 5 roda na sessão interativa do usuário e o IPC necessita de acesso ao diretório `Common\Files` do perfil do usuário, configure o serviço previamente instalado para rodar sob a mesma conta do operador:
```powershell
nssm set XauBot ObjectName ".\Administrator" "SuaSenhaSeguraAqui"
```
> [!NOTE]
> O serviço precisa estar instalado previamente (Passo 1) para que `nssm set XauBot ObjectName` seja aplicado com sucesso.

#### Passo 3: Iniciar o Serviço
```powershell
pwsh scripts\service-nssm.ps1 -Action Start
```

#### Passo 4: Consultar Status e Health Probe
```powershell
pwsh scripts\service-nssm.ps1 -Action Status
```

#### Passo 5: Inspecionar Logs em Tempo Real
```powershell
pwsh scripts\service-nssm.ps1 -Action Logs
```

#### Manutenção: Reiniciar e Parar
```powershell
# Reiniciar o serviço
pwsh scripts\service-nssm.ps1 -Action Restart

# Parar o serviço
pwsh scripts\service-nssm.ps1 -Action Stop
```

---

## 10. Observabilidade, Health Check e Alertas

### 10.1. Matriz de Códigos HTTP do Endpoint `/health`

| Resposta HTTP | Payload JSON | Significado Operacional | Ação Necessária |
| :--- | :--- | :--- | :--- |
| **HTTP 200 OK** | `{"status": "ok", "broker_connected": true, "trading_halted": false}` | Sistema 100% saudável e operando. | Nenhuma. Operação normal. |
| **HTTP 200 OK** | `{"status": "halted", "broker_connected": true, "trading_halted": true, "reasons": ["TRADING_HALTED"]}` | Salvaguarda de negócio ativa (Kill Switch diário ou pausa manual). Infraestrutura íntegra. | Não reiniciar o serviço. Aguardar próximo dia ou intervir manualmente se planejado. |
| **HTTP 503 Svc Unavail** | `{"status": "degraded", "broker_connected": false, "reasons": ["BROKER_DISCONNECTED"]}` | Falha de infraestrutura: terminal fechado ou desconectado. | O Watchdog tentará reconectar. Se persistir > 2 min, verificar terminal MT5. |

### 10.2. Monitoramento Externo com Uptime Kuma / Prometheus
Configure um monitor HTTP no Uptime Kuma ou similar:
- **URL:** `http://127.0.0.1:8000/health` (via VPN/Túnel seguro ou agente local)
- **Status esperados:** `200`
- **Intervalo:** 30 segundos
- **Alerta de Degradação:** Disparar se status retornar `503` ou timeout por mais de 2 checagens consecutivas.

---

## 11. Checklist Operacional Diário

### Pré-Abertura da Sessão de Londres (02:45 EST / 06:45 UTC)
- [ ] Acessar a VPS via RDP.
- [ ] Conferir se o relógio da VPS está sincronizado (`w32tm /query /status`).
- [ ] Executar verificação pré-voo com variáveis de ambiente carregadas da Seção 8: `pwsh scripts\verify-vps.ps1 -Detailed`.
- [ ] Conferir se o terminal MT5 está aberto e conectado (ícone de tráfego verde no canto inferior direito).
- [ ] Conferir se o gráfico M1 tem o EA `XauGuardian` anexado com carinha feliz / chapéu azul.
- [ ] Rodar `curl.exe http://127.0.0.1:8000/health` e validar retorno `status: "ok"`.
- [ ] Confirmar ausência de arquivos residuais de kill switch em `Common\Files`.


### Pós-Fechamento da Sessão de Nova York (17:00 EST / 21:00 UTC)
- [ ] Conferir o status do dia: `curl.exe http://127.0.0.1:8000/status`.
- [ ] Verificar relatório diário emitido no canal do Telegram.
- [ ] Confirmar que não há posições órfãs abertas no terminal.

### Procedimento de Retomada após Acionamento de Kill Switch
Caso o robô tenha sido paralisado por limite diário atingido:
1. Analisar os motivos no Telegram e nos logs em `logs\service_stdout.log`.
2. Não forçar reabertura no mesmo pregão.
3. Para o dia seguinte:
   - Excluir o arquivo `xau_killswitch.json` da pasta `Common\Files`.
   - Reiniciar o serviço: `pwsh scripts\service-nssm.ps1 -Action Restart`.
   - Confirmar que `/health` retornou ao estado `status: "ok"`.

---

## 12. Plano de Resposta a Desastres (DRP)

| Cenário de Falha | Comportamento Automático | Ação do Operador Humano |
| :--- | :--- | :--- |
| **Queda total da VPS** | Ordens protegidas por Stop Loss no servidor da corretora. | Acessar painel do provedor de VPS e reiniciar instância. Conferir ordens via app MT5 Mobile. |
| **Processo Python cai** | EA Guardião detecta silêncio > 30s, fecha ordens do bot a mercado. NSSM reinicia o Python em 5s. | Inspecionar `logs\service_stderr.log` para identificar a causa raiz. |
| **Terminal MT5 congela** | Watchdog do Python detecta tick stale (> 60s) e marca `/health` como 503. | Finalizar processo `terminal64.exe` via Task Manager; reabri-lo manualmente via RDP ou através de tarefa agendada / monitor de processo. |
| **Perda diária atinge 3%** | EA fecha posições imediatamente, grava kill switch; Python trava novos envios. | Nenhuma ação necessária. Deixar o sistema pausado até o próximo dia útil. |
