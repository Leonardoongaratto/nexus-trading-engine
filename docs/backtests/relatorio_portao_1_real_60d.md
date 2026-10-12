# Relatório de Análise Estatística de Backtest

## 1. Desempenho da Estratégia
- **Saldo Inicial**: US$ 10000.00
- **Saldo Final**: US$ 9984.08
- **Lucro Líquido**: US$ -15.92
- **Total de Trades**: 4
- **Taxa de Acerto**: 25.0%
- **Fator de Lucro**: 0.85
- **Expectativa em R**: -1.70R
- **Drawdown Máximo**: US$ 199.52 (1.96%)
- **Índice Sharpe Anualizado**: -0.30

## 2. Simulação de Monte Carlo
- **Simulações**: 200
- **Probabilidade de Ruína (DD Limiar)**: 0.00%
- **Drawdown Máximo Mediano**: US$ 91.20 (0.91%)
- **Drawdown Máximo P95**: US$ 155.36 (1.55%)
- **Drawdown Máximo P99**: US$ 159.52 (1.60%)
- **Saldo Final Mediano**: US$ 9984.08
- **Intervalo de Confiança 90% (P5 - P95)**: US$ 9844.64 - US$ 10128.16

## 3. Comparativo com Baseline Aleatório
- **Simulações Aleatórias**: 20
- **Expectativa Estratégia**: -1.70R
- **Fator de Lucro Estratégia**: 0.85
- **Expectativa Média Baseline**: -1.56R (+/- 0.75)
- **Fator de Lucro Médio Baseline**: 0.00
- **Taxa de Acerto Média Baseline**: 0.0%
- **p-valor Expectativa**: 0.6667
- **p-valor Fator de Lucro**: 0.0476
- **Estatisticamente Significativo**: NÃO (p >= 0.05)

## 4. Análise Walk-Forward
- **Janelas Analisadas**: 2
- **Eficiência Walk-Forward Geral (WFE)**: 0.00
- **Trades Fora da Amostra (OOS)**: 1
- **PnL Líquido OOS**: US$ -19.76
- **Fator de Lucro OOS**: 0.00
- **Drawdown Máximo OOS**: US$ 19.76 (0.20%)

## 5. Parecer do Portão de Decisão Humana 1
- **Amostra Mínima (>= 200 trades)**: REPROVADO (1 trades fora da amostra (OOS))
- **Lucratividade Líquida (PnL > 0, PF > 1.0, Exp > 0R)**: REPROVADO (US$ -19.76, PF 0.00, Exp -0.32R)
- **Superioridade vs Baseline (p < 0.05)**: REPROVADO
- **Eficiência Walk-Forward (WFE >= 0.50)**: REPROVADO (0.00)
- **Controle de Risco de Cauda (Ruína <= 1%, DD P99 <= 20%)**: APROVADO (Ruína 0.0%, DD P99 1.6%)
- **Veredito Proposto**: [AMOSTRA_INSUFICIENTE] PENDENTE: Amostra com 1 trades fora da amostra (OOS) (< 200 trades mínimos recomendados para o Portão 1).
