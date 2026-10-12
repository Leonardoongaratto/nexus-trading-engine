# EA Guardião MQL5 (`XauGuardian.mq5`)

## 1. Visão Geral e Arquitetura

O `XauGuardian.mq5` é um Expert Advisor desenvolvido em MQL5 para atuar como a **segunda linha de defesa independente** do robô `xau-bot` diretamente dentro do MetaTrader 5 (`terminal64.exe`).

Ele roda isoladamente em uma thread própria do terminal (geralmente anexado a um gráfico exclusivo, como M1/M5 de qualquer par ou do próprio XAU/USD) e opera através do diretório de arquivos comuns do MetaTrader 5 (`FILE_COMMON`, `%APPDATA%\MetaQuotes\Terminal\Common\Files`).

```
                ┌───────────────── VPS Windows (perto da corretora) ─────────────────┐
                │                                                                      │
 corretora ◄───►│ terminal64.exe ◄─IPC─► xau-bot (Python 3.12, serviço NSSM)          │
                │   └─ EA guardião        ├─ mt5_adapter (1 thread, única porta)       │
                │      · heartbeat ◄──────┤─ execution (retcodes, idempotência)        │
                │      · kill switch ────►│─ runtime (watchdog, runner)                │
                │      · calendário ──────►├─ filters (NewsFilter, staleness check)     │
                │        (Common\Files)    ├─ /health /status (127.0.0.1)              │
                │                          └─ alertas (Telegram)                      │
                └──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Responsabilidades Principais

1. **Monitoramento de Heartbeat (Fail-Safe por Silêncio do Python)**:
   - Lê a cada 1 segundo (`OnTimer`) o arquivo `xau_heartbeat.json` emitido pelo Python com timestamp UTC.
   - Mede continuamente o silêncio real: `silence = TimeGMT() - ExtLastHeartbeatTime`.
   - Se o silêncio ultrapassar `InpHeartbeatTimeoutSec` (padrão 30s): assume que o processo Python travou, sofreu deadlock ou foi encerrado abruptamente.
   - Executa fechamento emergencial a mercado de todas as posições abertas com o `InpMagicNumber` e remove ordens pendentes.
   - Grava `xau_killswitch.json` com motivo `HEARTBEAT_TIMEOUT`.

2. **Kill Switch Mecânico Independente na Corretora**:
   - Calcula a cada ciclo a perda acumulada do dia (lucro/perda de deals fechados no histórico a partir das 00:00 + lucro flutuante atual de posições abertas).
   - Calcula o drawdown percentual da conta em relação ao **pico histórico de equity** (`peak_equity`), persistido na GlobalVariable do terminal `XAU_GUARDIAN_PEAK_<login>_<magic>`, isolado por conta de negociação.
   - Se a perda diária (`InpMaxDailyLossPct`, padrão 3.0%) ou o drawdown da conta (`InpMaxDrawdownPct`, padrão 10.0%) for atingido:
     - Fecha todas as posições abertas associadas ao MagicNumber.
     - Cancela todas as ordens pendentes.
     - Grava atomicamente `xau_killswitch.json` com `DAILY_LOSS_BREACHED` ou `DRAWDOWN_BREACHED`.
     - O Python detecta esse arquivo e sincroniza o `RiskGuard` (`trading_halted=True`).

3. **Exportação Automática do Calendário Econômico Nativo**:
   - Extrai eventos econômicos nativos do MetaTrader 5 (`CalendarValueHistory`) com janela de lookahead configurável (padrão 7 dias).
   - Filtra eventos relevantes de moedas-chave (`USD`, `EUR`, `GBP`) com impacto Médio e Alto.
   - Converte datas para GMT UTC real (`Z`), corrige escala ($10^6$) e grava em UTF-8 padronizado (`calendar.json`) com metadado `generated_at`.
   - O `NewsFilter` do Python ingere esse arquivo diretamente, validando frescor e evitando a necessidade de APIs pagas de notícias.

---

## 3. Protocolo de Arquivos Compartilhados (`Common\Files`)

Os arquivos são gravados no diretório compartilhado do MetaTrader 5:
- **Windows**: `%APPDATA%\MetaQuotes\Terminal\Common\Files`
- **Variável de Ambiente Python**: `XAU_GUARDIAN_COMMON_DIR`

| Arquivo | Origem | Destino | Função |
| :--- | :--- | :--- | :--- |
| `xau_heartbeat.json` | Python (`HeartbeatWriter`) | EA Guardião MQL5 | Pulso contínuo com timestamp ISO UTC, PID, sequência monotônica e estado. |
| `xau_killswitch.json` | EA Guardião MQL5 | Python (`GuardianBridge`) | Notificação de emergência com razão do corte, PnL e contagem de ordens encerradas. |
| `calendar.json` | EA Guardião MQL5 | Python (`NewsFilter`) | Eventos do calendário econômico nativo do MT5 em UTF-8 com `generated_at`. |

---

## 4. Parâmetros de Entrada (`Inputs`)

| Parâmetro | Tipo | Padrão | Descrição |
| :--- | :--- | :--- | :--- |
| `InpMagicNumber` | `ulong` | `202610` | Magic Number utilizado pelo bot Python para identificar ordens. |
| `InpHeartbeatTimeoutSec` | `int` | `30` | Segundos máximos de tolerância sem novo heartbeat do Python. |
| `InpClosePositionsOnTimeout` | `bool` | `true` | Fechar posições a mercado se o heartbeat expirar. |
| `InpDeleteOrdersOnTimeout` | `bool` | `true` | Excluir ordens pendentes se o heartbeat expirar. |
| `InpHeartbeatFileName` | `string` | `"xau_heartbeat.json"` | Nome do arquivo de heartbeat em `Common\Files`. |
| `InpMaxDailyLossPct` | `double` | `3.0` | Limite de perda diária permitida (% do saldo do início do dia). |
| `InpMaxDrawdownPct` | `double` | `10.0` | Limite de drawdown máximo da conta (% do pico histórico de equity). |
| `InpCloseOnKillSwitch` | `bool` | `true` | Fechar posições se o kill switch for acionado. |
| `InpDeleteOnKillSwitch` | `bool` | `true` | Cancelar ordens se o kill switch for acionado. |
| `InpKillSwitchFileName` | `string` | `"xau_killswitch.json"` | Nome do arquivo de sinal em `Common\Files`. |
| `InpExportCalendar` | `bool` | `true` | Ativar exportação do calendário econômico do terminal. |
| `InpCalendarExportIntervalMin` | `int` | `60` | Frequência de atualização do calendário (minutos). |
| `InpCalendarFileName` | `string` | `"calendar.json"` | Nome do arquivo de calendário em `Common\Files`. |
| `InpCalendarDaysLookahead` | `int` | `7` | Horizonte de dias futuros a exportar. |

---

## 5. Instruções de Instalação e Compilação

1. Abra o MetaTrader 5 e pressione `F4` para abrir o **MetaEditor**.
2. No painel de navegação à esquerda, clique com o botão direito em `Experts` $\to$ `Abrir Pasta`.
3. Copie o arquivo `XauGuardian.mq5` para a pasta `MQL5\Experts\`.
4. No MetaEditor, abra `XauGuardian.mq5` e clique em **Compilar** (`F7`).
5. Certifique-se de que a compilação gerou `0 errors, 0 warnings`.
6. No terminal MetaTrader 5, abra um gráfico qualquer (ex: `XAUUSD, M5`).
7. Arraste o `XauGuardian` do Navegador para o gráfico.
8. Na aba "Comum", marque **Permitir Algo Trading** e clique em OK.

---

## 6. Procedimento Operacional de Retomada e Reset

### Retomada após Kill Switch
Se o kill switch mecânico ou o fail-safe de silêncio disparar, o arquivo `xau_killswitch.json` é gravado no diretório `Common\Files`.
Para retomar as operações após análise do operador humano:
1. Feche ou ajuste as posições pendentes manualmente.
2. Apague o arquivo `xau_killswitch.json` na pasta `%APPDATA%\MetaQuotes\Terminal\Common\Files`.
3. Reinicie o bot Python ou envie comando de liberação. O Python e o EA detectam a ausência do sinal de corte e liberam as operações no próximo ciclo.

### Reset do Pico de Equity
O EA persiste o pico de equity da conta em uma variável global do terminal chamada `XAU_GUARDIAN_PEAK_<login>_<magic>`.
Em caso de depósitos/saques na conta que alterem a base de capital:
1. No MetaTrader 5, pressione a tecla `F3` (Variáveis Globais).
2. Localize a entrada `XAU_GUARDIAN_PEAK_<login>_<magic>`.
3. Pressione a tecla `Delete` para removê-la. O EA recalculará o pico no próximo tick a partir do equity atual da conta.
