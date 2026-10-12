//+------------------------------------------------------------------+
//|                                                  XauGuardian.mq5 |
//|                                  Copyright 2026, xau-bot project |
//|                   https://github.com/Leonardoongaratto/ai-dev-template |
//+------------------------------------------------------------------+
#property copyright   "Copyright 2026, xau-bot project"
#property link        "https://github.com/Leonardoongaratto/ai-dev-template"
#property version     "1.30"
#property description "Expert Advisor Guardiao MQL5 para XAU/USD: Heartbeat fail-safe, kill switch independente e exportador de calendario"

#include <Trade\Trade.mqh>

//--- Parametros de Entrada
input group "--- Heartbeat Safeguard ---"
input ulong    InpMagicNumber               = 202610;               // Magic Number do xau-bot
input int      InpHeartbeatTimeoutSec       = 30;                   // Timeout sem Heartbeat (segundos)
input bool     InpClosePositionsOnTimeout   = true;                 // Fechar posicoes no timeout
input bool     InpDeleteOrdersOnTimeout     = true;                 // Cancelar ordens pendentes no timeout
input string   InpHeartbeatFileName         = "xau_heartbeat.json"; // Nome do arquivo de Heartbeat (Common)

input group "--- Kill Switch Salvaguarda Independente ---"
input double   InpMaxDailyLossPct           = 3.0;                  // Perda diaria maxima (% do saldo inicial)
input double   InpMaxDrawdownPct            = 10.0;                 // Drawdown maximo (% do pico de equity da conta)
input bool     InpCloseOnKillSwitch         = true;                 // Fechar posicoes ao acionar Kill Switch
input bool     InpDeleteOnKillSwitch        = true;                 // Cancelar ordens ao acionar Kill Switch
input string   InpKillSwitchFileName        = "xau_killswitch.json";// Nome do arquivo de Kill Switch (Common)

input group "--- Exportacao do Calendario Economico ---"
input bool     InpExportCalendar            = true;                 // Exportar calendario economico nativo
input int      InpCalendarExportIntervalMin = 60;                   // Intervalo de exportacao (minutos)
input string   InpCalendarFileName          = "calendar.json";      // Nome do arquivo de calendario (Common)
input int      InpCalendarDaysLookahead     = 7;                    // Dias a frente para exportacao

//--- Variaveis Globais de Estado
CTrade         ExtTrade;
datetime       ExtLastHeartbeatTime         = 0;
datetime       ExtLastCalendarExport        = 0;
bool           ExtEmergencyHalted           = false;
bool           ExtKillSwitchActive          = false;
double         ExtStartDayBalance           = 0.0;
datetime       ExtCurrentTradingDay         = 0;

//+------------------------------------------------------------------+
//| Formata datetime para string ISO 8601 UTC                        |
//+------------------------------------------------------------------+
string TimeToIsoUtc(datetime t)
{
   MqlDateTime dt;
   TimeToStruct(t, dt);
   return StringFormat("%04d-%02d-%02dT%02d:%02d:%02dZ",
                       dt.year, dt.mon, dt.day, dt.hour, dt.min, dt.sec);
}

//+------------------------------------------------------------------+
//| Formata valor de calendario do MT5 (escala 10^6 e LONG_MIN)      |
//+------------------------------------------------------------------+
string FormatCalendarValue(long raw_val)
{
   if(raw_val == LONG_MIN || raw_val == WRONG_VALUE)
      return "null";
   double val = (double)raw_val / 1000000.0;
   return DoubleToString(val, 2);
}

//+------------------------------------------------------------------+
//| Extrai valor de string entre aspas apos chave JSON               |
//+------------------------------------------------------------------+
string ExtractJsonString(const string json, const string key)
{
   string pattern = "\"" + key + "\":";
   int pos = StringFind(json, pattern);
   if(pos < 0) return "";
   
   int start = pos + StringLen(pattern);
   while(start < StringLen(json) && (StringGetCharacter(json, start) == ' ' || StringGetCharacter(json, start) == '\"'))
      start++;
      
   int end = start;
   while(end < StringLen(json) && StringGetCharacter(json, end) != '\"' && StringGetCharacter(json, end) != ',' && StringGetCharacter(json, end) != '}')
      end++;
      
   return StringSubstr(json, start, end - start);
}

//+------------------------------------------------------------------+
//| Le e analisa o arquivo de heartbeat no Common Files              |
//+------------------------------------------------------------------+
bool ReadHeartbeat(datetime &out_time, string &out_status, bool &out_halted, int &out_seq)
{
   if(!FileIsExist(InpHeartbeatFileName, FILE_COMMON))
      return false;
      
   int handle = FileOpen(InpHeartbeatFileName, FILE_READ | FILE_COMMON | FILE_TXT | FILE_ANSI | FILE_SHARE_READ, 0, CP_UTF8);
   if(handle == INVALID_HANDLE)
      return false;
      
   string content = "";
   while(!FileIsEnding(handle))
   {
      content += FileReadString(handle);
   }
   FileClose(handle);
   
   if(StringLen(content) < 10)
      return false;
      
   string timeStr = ExtractJsonString(content, "timestamp");
   out_status     = ExtractJsonString(content, "status");
   string haltStr = ExtractJsonString(content, "trading_halted");
   string seqStr  = ExtractJsonString(content, "sequence");
   
   out_halted = (haltStr == "true");
   out_seq    = (int)StringToInteger(seqStr);
   
   if(StringLen(timeStr) >= 19)
   {
      string formatted = StringSubstr(timeStr, 0, 10) + " " + StringSubstr(timeStr, 11, 8);
      StringReplace(formatted, "-", ".");
      out_time = StringToTime(formatted);
      if(out_time > 0)
         return true;
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Fecha todas as posicoes abertas associadas ao MagicNumber        |
//+------------------------------------------------------------------+
int ClosePositionsByMagic(ulong magic)
{
   int closed = 0;
   int total = PositionsTotal();
   for(int i = total - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0)
      {
         if((ulong)PositionGetInteger(POSITION_MAGIC) == magic)
         {
            if(ExtTrade.PositionClose(ticket))
               closed++;
         }
      }
   }
   return closed;
}

//+------------------------------------------------------------------+
//| Cancela todas as ordens pendentes associadas ao MagicNumber      |
//+------------------------------------------------------------------+
int DeleteOrdersByMagic(ulong magic)
{
   int cancelled = 0;
   int total = OrdersTotal();
   for(int i = total - 1; i >= 0; i--)
   {
      ulong ticket = OrderGetTicket(i);
      if(ticket > 0)
      {
         if((ulong)OrderGetInteger(ORDER_MAGIC) == magic)
         {
            if(ExtTrade.OrderDelete(ticket))
               cancelled++;
         }
      }
   }
   return cancelled;
}

//+------------------------------------------------------------------+
//| Grava sinal de Kill Switch no Common Files de forma atomica      |
//+------------------------------------------------------------------+
bool WriteKillSwitchSignal(string reason, double current_loss_pct, double max_limit_pct, int closed, int cancelled)
{
   string tmp_file = InpKillSwitchFileName + ".tmp";
   int handle = FileOpen(tmp_file, FILE_WRITE | FILE_COMMON | FILE_TXT | FILE_ANSI, 0, CP_UTF8);
   if(handle == INVALID_HANDLE)
   {
      PrintFormat("[XauGuardian] ERRO: Nao foi possivel criar arquivo temporario %s", tmp_file);
      return false;
   }
   
   string json = StringFormat(
      "{\n"
      "  \"timestamp\": \"%s\",\n"
      "  \"reason\": \"%s\",\n"
      "  \"current_loss_pct\": %.2f,\n"
      "  \"max_loss_pct\": %.2f,\n"
      "  \"positions_closed\": %d,\n"
      "  \"orders_cancelled\": %d,\n"
      "  \"details\": \"Acionado pelo EA Guardiao nativo MQL5\"\n"
      "}\n",
      TimeToIsoUtc(TimeGMT()),
      reason,
      current_loss_pct,
      max_limit_pct,
      closed,
      cancelled
   );
   
   FileWriteString(handle, json);
   FileClose(handle);
   
   if(!FileMove(tmp_file, FILE_COMMON, InpKillSwitchFileName, FILE_COMMON | FILE_REWRITE))
   {
      PrintFormat("[XauGuardian] ERRO: Falha ao mover %s para %s", tmp_file, InpKillSwitchFileName);
      return false;
   }
   
   PrintFormat("[XauGuardian] SINAL DE KILL SWITCH GRAVADO: %s (loss: %.2f%%, limit: %.2f%%, closed: %d, cancelled: %d)",
               reason, current_loss_pct, max_limit_pct, closed, cancelled);
   return true;
}

//+------------------------------------------------------------------+
//| Exporta eventos do Calendario Economico Nativo em JSON UTF-8     |
//+------------------------------------------------------------------+
void ExportEconomicCalendar()
{
   if(!InpExportCalendar) return;
   
   datetime now_server = TimeTradeServer();
   datetime fromTime = now_server - 86400;
   datetime toTime   = now_server + (InpCalendarDaysLookahead * 86400);
   
   MqlCalendarValue values[];
   int count = CalendarValueHistory(values, fromTime, toTime);
   if(count <= 0)
   {
      ExtLastCalendarExport = TimeGMT() - ((InpCalendarExportIntervalMin * 60) - 60);
      return;
   }
      
   string tmp_file = InpCalendarFileName + ".tmp";
   int handle = FileOpen(tmp_file, FILE_WRITE | FILE_COMMON | FILE_TXT | FILE_ANSI, 0, CP_UTF8);
   if(handle == INVALID_HANDLE)
   {
      ExtLastCalendarExport = TimeGMT() - ((InpCalendarExportIntervalMin * 60) - 60);
      return;
   }
      
   FileWriteString(handle, "{\n");
   FileWriteString(handle, StringFormat("  \"generated_at\": \"%s\",\n", TimeToIsoUtc(TimeGMT())));
   FileWriteString(handle, "  \"events\": [\n");
   
   long gmt_offset_sec = (long)(TimeTradeServer() - TimeGMT());
   int written = 0;
   for(int i = 0; i < count; i++)
   {
      MqlCalendarEvent event_info;
      if(!CalendarEventById(values[i].event_id, event_info))
         continue;
         
      MqlCalendarCountry country_info;
      CalendarCountryById(event_info.country_id, country_info);
      string currency = country_info.currency;
      
      if(currency != "USD" && currency != "EUR" && currency != "GBP")
         continue;
         
      string impact = "LOW";
      if(event_info.importance == CALENDAR_IMPORTANCE_HIGH)
         impact = "HIGH";
      else if(event_info.importance == CALENDAR_IMPORTANCE_MODERATE)
         impact = "MEDIUM";
      else
         continue;
         
      if(written > 0)
         FileWriteString(handle, ",\n");
         
      string safe_title = event_info.name;
      StringReplace(safe_title, "\"", "'");
      StringReplace(safe_title, "\\", "/");
      
      string actualStr   = FormatCalendarValue(values[i].actual_value);
      string forecastStr = FormatCalendarValue(values[i].forecast_value);
      string prevStr     = FormatCalendarValue(values[i].prev_value);
      
      datetime event_gmt = values[i].time - (datetime)gmt_offset_sec;
      
      string eventJson = StringFormat(
         "    {\n"
         "      \"title\": \"%s\",\n"
         "      \"currency\": \"%s\",\n"
         "      \"impact\": \"%s\",\n"
         "      \"time\": \"%s\",\n"
         "      \"actual\": %s,\n"
         "      \"forecast\": %s,\n"
         "      \"previous\": %s,\n"
         "      \"source_id\": \"%I64u\"\n"
         "    }",
         safe_title,
         currency,
         impact,
         TimeToIsoUtc(event_gmt),
         actualStr,
         forecastStr,
         prevStr,
         values[i].id
      );
      
      FileWriteString(handle, eventJson);
      written++;
   }
   
   FileWriteString(handle, "\n  ]\n}\n");
   FileClose(handle);
   
   if(FileMove(tmp_file, FILE_COMMON, InpCalendarFileName, FILE_COMMON | FILE_REWRITE))
   {
      ExtLastCalendarExport = TimeGMT();
      PrintFormat("[XauGuardian] Calendario economico exportado com sucesso (%d eventos)", written);
   }
   else
   {
      PrintFormat("[XauGuardian] ERRO ao mover arquivo temporario %s para %s", tmp_file, InpCalendarFileName);
      ExtLastCalendarExport = TimeGMT() - ((InpCalendarExportIntervalMin * 60) - 60);
   }
}

//+------------------------------------------------------------------+
//| Calcula a perda diaria acumulada e drawdown atual                |
//+------------------------------------------------------------------+
void EvaluateDailyRiskLimits(double &loss_pct, double &drawdown_pct)
{
   datetime now = TimeCurrent();
   MqlDateTime dt;
   TimeToStruct(now, dt);
   dt.hour = 0; dt.min = 0; dt.sec = 0;
   datetime day_start = StructToTime(dt);
   
   if(ExtCurrentTradingDay != day_start)
   {
      ExtCurrentTradingDay = day_start;
      ExtStartDayBalance = AccountInfoDouble(ACCOUNT_BALANCE);
      if(!FileIsExist(InpKillSwitchFileName, FILE_COMMON))
      {
         ExtKillSwitchActive = false;
      }
   }
   
   if(ExtStartDayBalance <= 0.0)
      ExtStartDayBalance = AccountInfoDouble(ACCOUNT_BALANCE);
      
   if(ExtStartDayBalance <= 0.0)
   {
      loss_pct = 0.0;
      drawdown_pct = 0.0;
      return;
   }
   
   // 1. Calcular lucro realizado do dia a partir do historico de deals
   HistorySelect(day_start, now);
   int total_deals = HistoryDealsTotal();
   double realized_today = 0.0;
   for(int i = 0; i < total_deals; i++)
   {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket > 0)
      {
         long entry = HistoryDealGetInteger(ticket, DEAL_ENTRY);
         if(entry == DEAL_ENTRY_OUT || entry == DEAL_ENTRY_INOUT || entry == DEAL_ENTRY_OUT_BY)
         {
            realized_today += HistoryDealGetDouble(ticket, DEAL_PROFIT)
                            + HistoryDealGetDouble(ticket, DEAL_SWAP)
                            + HistoryDealGetDouble(ticket, DEAL_COMMISSION);
         }
      }
   }
   
   // 2. Lucro nao realizado (flutuante) das posicoes abertas
   double floating = AccountInfoDouble(ACCOUNT_EQUITY) - AccountInfoDouble(ACCOUNT_BALANCE);
   double total_day_pnl = realized_today + floating;
   
   if(total_day_pnl < 0.0)
      loss_pct = (MathAbs(total_day_pnl) / ExtStartDayBalance) * 100.0;
   else
      loss_pct = 0.0;
      
   // 3. Drawdown da conta a partir do pico historico de equity (isolado por conta)
   long account_login = AccountInfoInteger(ACCOUNT_LOGIN);
   string peak_var_name = StringFormat("XAU_GUARDIAN_PEAK_%I64d_%I64u", account_login, InpMagicNumber);
   double peak_equity = 0.0;
   if(GlobalVariableCheck(peak_var_name))
      peak_equity = GlobalVariableGet(peak_var_name);
      
   double current_equity = AccountInfoDouble(ACCOUNT_EQUITY);
   if(current_equity > peak_equity)
   {
      peak_equity = current_equity;
      GlobalVariableSet(peak_var_name, peak_equity);
   }
   
   if(peak_equity > 0.0 && current_equity < peak_equity)
      drawdown_pct = ((peak_equity - current_equity) / peak_equity) * 100.0;
   else
      drawdown_pct = 0.0;
}

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   ExtTrade.SetExpertMagicNumber(InpMagicNumber);
   ExtTrade.SetDeviationInPoints(20);
   
   ExtEmergencyHalted   = false;
   ExtKillSwitchActive  = false;
   ExtLastHeartbeatTime = TimeGMT();
   ExtLastCalendarExport= 0;
   
   datetime now = TimeCurrent();
   MqlDateTime dt;
   TimeToStruct(now, dt);
   dt.hour = 0; dt.min = 0; dt.sec = 0;
   ExtCurrentTradingDay = StructToTime(dt);
   ExtStartDayBalance   = AccountInfoDouble(ACCOUNT_BALANCE);
   
   if(!EventSetTimer(1))
   {
      Print("[XauGuardian] ERRO: Falha ao inicializar EventSetTimer");
      return INIT_FAILED;
   }
   
   PrintFormat("[XauGuardian] Guardiao MQL5 Inicializado com sucesso. Login: %I64d, Magic: %I64u, Timeout: %ds, MaxLoss: %.1f%%, MaxDD: %.1f%%",
               AccountInfoInteger(ACCOUNT_LOGIN), InpMagicNumber, InpHeartbeatTimeoutSec, InpMaxDailyLossPct, InpMaxDrawdownPct);
               
   ExportEconomicCalendar();
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   PrintFormat("[XauGuardian] Guardiao MQL5 finalizado. Razao: %d", reason);
}

//+------------------------------------------------------------------+
//| Expert timer function                                            |
//+------------------------------------------------------------------+
void OnTimer()
{
   datetime hb_time = 0;
   string status = "";
   bool halted = false;
   int seq = 0;
   
   // 1. Checagem de Heartbeat e Medicao Rigorosa de Silencio
   datetime now_gmt = TimeGMT();
   bool read_ok = ReadHeartbeat(hb_time, status, halted, seq);
   if(read_ok)
   {
      ExtLastHeartbeatTime = hb_time;
   }
   
   int silence_sec = (int)(now_gmt - ExtLastHeartbeatTime);
   if(silence_sec > InpHeartbeatTimeoutSec)
   {
      if(!ExtEmergencyHalted)
      {
         ExtEmergencyHalted = true;
         PrintFormat("[XauGuardian] ALERTA CRITICO: Heartbeat expirou! Silencio de %d s > %d s",
                     silence_sec, InpHeartbeatTimeoutSec);
                     
         int closed = 0;
         int cancelled = 0;
         if(InpClosePositionsOnTimeout)
            closed = ClosePositionsByMagic(InpMagicNumber);
         if(InpDeleteOrdersOnTimeout)
            cancelled = DeleteOrdersByMagic(InpMagicNumber);
            
         WriteKillSwitchSignal("HEARTBEAT_TIMEOUT", 0.0, 0.0, closed, cancelled);
      }
   }
   else if(read_ok && ExtEmergencyHalted)
   {
      PrintFormat("[XauGuardian] Heartbeat restaurado pelo Python (silencio: %ds). Recuperando estado.", silence_sec);
      ExtEmergencyHalted = false;
   }
   
   // 2. Checagem de Limite de Risco e Kill Switch Mecanico Independente
   double loss_pct = 0.0;
   double drawdown_pct = 0.0;
   EvaluateDailyRiskLimits(loss_pct, drawdown_pct);
   
   if((loss_pct >= InpMaxDailyLossPct || drawdown_pct >= InpMaxDrawdownPct) && !ExtKillSwitchActive)
   {
      ExtKillSwitchActive = true;
      string reason = (loss_pct >= InpMaxDailyLossPct) ? "DAILY_LOSS_BREACHED" : "DRAWDOWN_BREACHED";
      double active_pct = (loss_pct >= InpMaxDailyLossPct) ? loss_pct : drawdown_pct;
      double limit_pct  = (loss_pct >= InpMaxDailyLossPct) ? InpMaxDailyLossPct : InpMaxDrawdownPct;
      
      PrintFormat("[XauGuardian] KILL SWITCH MECANICO ACIONADO! %s (atual: %.2f%%, limite: %.2f%%)",
                  reason, active_pct, limit_pct);
      
      int closed = 0;
      int cancelled = 0;
      if(InpCloseOnKillSwitch)
         closed = ClosePositionsByMagic(InpMagicNumber);
      if(InpDeleteOnKillSwitch)
         cancelled = DeleteOrdersByMagic(InpMagicNumber);
         
      WriteKillSwitchSignal(reason, active_pct, limit_pct, closed, cancelled);
   }
   else if(ExtKillSwitchActive && !FileIsExist(InpKillSwitchFileName, FILE_COMMON) && loss_pct < InpMaxDailyLossPct && drawdown_pct < InpMaxDrawdownPct)
   {
      Print("[XauGuardian] Kill switch rearmado pelo operador (arquivo removido e risco sob controle). Operacoes liberadas.");
      ExtKillSwitchActive = false;
   }
   
   // 3. Exportacao Periodica do Calendario Economico
   if(InpExportCalendar)
   {
      int elapsed_min = (int)((TimeGMT() - ExtLastCalendarExport) / 60);
      if(elapsed_min >= InpCalendarExportIntervalMin || ExtLastCalendarExport == 0)
      {
         ExportEconomicCalendar();
      }
   }
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   if(ExtKillSwitchActive || ExtEmergencyHalted)
   {
      if(InpDeleteOnKillSwitch || InpDeleteOrdersOnTimeout)
      {
         if(OrdersTotal() > 0)
            DeleteOrdersByMagic(InpMagicNumber);
      }
   }
}
//+------------------------------------------------------------------+
