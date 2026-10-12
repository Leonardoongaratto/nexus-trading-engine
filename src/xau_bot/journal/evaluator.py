"""Evaluation engine for trade outcomes, excursions and loss taxonomy."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from xau_bot.advisor.modes import ProposalStatus, TradeProposal
from xau_bot.domain.models import Candle, Direction, SwingType
from xau_bot.filters.sessions import SessionFilter
from xau_bot.journal.models import (
    JournalEntry,
    LossCategory,
    LossRootCause,
    OutcomeData,
    SnapshotData,
    TradeOutcome,
)


def infer_session(dt: datetime) -> str:
    """Infers market session name using canonical timezone boundaries."""
    sf = SessionFilter()
    ny_time = dt.astimezone(sf.ny_tz)
    london_time = dt.astimezone(sf.london_tz)

    if 8 <= ny_time.hour < 17:
        return "NY"

    if 7 <= london_time.hour < 17:
        return "LONDON"

    return "ASIA"


def create_snapshot_from_proposal(
    proposal: TradeProposal,
    session: str | None = None,
    news_distance_min: float | None = None,
    spread_price: float | None = None,
    htf_bias: str | None = None,
    displacement_atr: float | None = None,
    fvg_size_atr: float | None = None,
    user_decision: str | None = None,
) -> SnapshotData:
    """Constructs a SnapshotData model from a live TradeProposal and its context."""
    setup = proposal.setup
    inferred_session = session or infer_session(proposal.created_at)

    sh_price = 0.0
    sl_price = 0.0
    if setup and setup.target_swing:
        if setup.target_swing.type == SwingType.HIGH:
            sh_price = setup.target_swing.price
        else:
            sl_price = setup.target_swing.price

    poi_name = setup.poi.__class__.__name__ if setup and setup.poi else "FVG"
    choch_lvl = setup.choch.price if setup and setup.choch else 0.0
    swept = (setup.sweep is not None) if setup else False
    atr_val = proposal.atr if proposal.atr is not None and proposal.atr > 0 else 1.0

    disp_atr = displacement_atr
    if disp_atr is None:
        if (
            setup
            and setup.poi
            and hasattr(setup.poi, "displacement_atr_ratio")
            and setup.poi.displacement_atr_ratio > 0
        ):
            disp_atr = round(float(setup.poi.displacement_atr_ratio), 2)
        elif setup and setup.dealing_range and atr_val > 0:
            disp_atr = round(setup.dealing_range.range / atr_val, 2)
        else:
            disp_atr = 1.5

    fvg_size = fvg_size_atr
    if fvg_size is None:
        if setup and setup.poi:
            if hasattr(setup.poi, "size") and atr_val > 0:
                fvg_size = round(float(setup.poi.size) / atr_val, 2)
            elif hasattr(setup.poi, "fvg") and setup.poi.fvg and atr_val > 0:
                fvg_size = round(float(setup.poi.fvg.size) / atr_val, 2)
        if fvg_size is None:
            fvg_size = 0.8

    bias = htf_bias
    if bias is None:
        if setup and setup.confluences:
            if "HTF_ALIGNED" in setup.confluences:
                bias = proposal.direction.value
            elif "HTF_COUNTER" in setup.confluences:
                bias = "BEARISH" if proposal.direction == Direction.BULLISH else "BULLISH"
        if bias is None:
            bias = proposal.direction.value

    spread_val = spread_price if spread_price is not None else 0.2

    decision = user_decision
    if decision is None:
        if proposal.status in (ProposalStatus.APPROVED, ProposalStatus.DISPATCHED):
            decision = "APPROVED"
        elif proposal.status == ProposalStatus.REJECTED:
            decision = "REJECTED"
        elif proposal.status == ProposalStatus.EXPIRED:
            decision = "EXPIRED"
        elif proposal.status == ProposalStatus.CANCELLED:
            decision = "CANCELLED"
        else:
            decision = "PENDING"

    return SnapshotData(
        timestamp=proposal.created_at,
        session=inferred_session,
        htf_bias=bias,
        swing_high=sh_price,
        swing_low=sl_price,
        poi_type=poi_name,
        poi_level=proposal.entry_price,
        swept_liquidity=swept,
        displacement_atr=disp_atr,
        fvg_size_atr=fvg_size,
        choch_level=choch_lvl,
        spread_price=spread_val,
        atr_m5=atr_val,
        news_distance_min=news_distance_min,
        user_decision=decision,
        user_reason=proposal.reason or "",
        confluences=setup.confluences if setup else (),
    )


class OutcomeEvaluator:
    """Calculates trade metrics (MFE, MAE, R-multiples) and diagnoses loss root causes."""

    def __init__(
        self,
        min_displacement_atr: float = 1.0,
        max_spread_atr_ratio: float = 0.35,
        min_sl_atr_ratio: float = 0.5,
        news_blackout_window_min: float = 30.0,
    ) -> None:
        self.min_displacement_atr = min_displacement_atr
        self.max_spread_atr_ratio = max_spread_atr_ratio
        self.min_sl_atr_ratio = min_sl_atr_ratio
        self.news_blackout_window_min = news_blackout_window_min

    def evaluate_position(
        self,
        entry: JournalEntry,
        exit_price: float,
        exit_time: datetime,
        position_candles: Sequence[Candle],
        subsequent_candles: Sequence[Candle] = (),
        open_time: datetime | None = None,
        fill_price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        pnl_usd: float = 0.0,
        exit_reason: str = "",
        slippage_points: float = 0.0,
    ) -> tuple[OutcomeData, TradeOutcome, LossRootCause | None, LossCategory | None]:
        entry_p = fill_price if fill_price is not None else entry.snapshot.poi_level
        sl_p = (
            stop_loss
            if stop_loss is not None
            else (entry.outcome_data.stop_loss if entry.outcome_data else 0.0)
        )
        tp_p = (
            take_profit
            if take_profit is not None
            else (entry.outcome_data.take_profit if entry.outcome_data else 0.0)
        )

        if sl_p == 0.0:
            risk_dist = max(1.0, entry.snapshot.atr_m5 * 1.5)
            sl_p = (
                entry_p - risk_dist if entry.direction == Direction.BULLISH else entry_p + risk_dist
            )
        else:
            risk_dist = abs(entry_p - sl_p)
            if risk_dist <= 1e-6:
                risk_dist = max(1.0, entry.snapshot.atr_m5)

        mfe_val = 0.0
        mae_val = 0.0

        for c in position_candles:
            if entry.direction == Direction.BULLISH:
                fav = c.high - entry_p
                adv = entry_p - c.low
            else:
                fav = entry_p - c.low
                adv = c.high - entry_p

            if fav > mfe_val:
                mfe_val = fav
            if adv > mae_val:
                mae_val = adv

        mfe_r = round(mfe_val / risk_dist, 2)
        mae_r = round(mae_val / risk_dist, 2)

        if entry.direction == Direction.BULLISH:
            realized_gain = exit_price - entry_p
        else:
            realized_gain = entry_p - exit_price

        realized_r = round(realized_gain / risk_dist, 2)
        max_r = max(mfe_r, realized_r)

        start_ts = open_time or entry.created_at
        duration = max(0.0, (exit_time - start_ts).total_seconds())

        post_exit = ""
        if subsequent_candles and tp_p > 0:
            tp_reached = False
            for sc in subsequent_candles:
                bull_tp = entry.direction == Direction.BULLISH and sc.high >= tp_p
                bear_tp = entry.direction == Direction.BEARISH and sc.low <= tp_p
                if bull_tp or bear_tp:
                    tp_reached = True
                    break
            if tp_reached:
                post_exit = "TP reached after exit"

        if realized_r >= 0.5:
            outcome = TradeOutcome.WIN
            loss_cause = None
            loss_cat = None
        elif -0.2 < realized_r < 0.5:
            outcome = TradeOutcome.BREAKEVEN
            loss_cause = None
            loss_cat = None
        else:
            outcome = TradeOutcome.LOSS
            loss_cause, loss_cat = self.diagnose_loss(
                entry.snapshot,
                risk_dist,
                exit_reason,
                direction=entry.direction,
            )

        outcome_data = OutcomeData(
            executed=True,
            entry_price=entry_p,
            stop_loss=sl_p,
            take_profit=tp_p,
            fill_price=entry_p,
            exit_price=exit_price,
            slippage_points=slippage_points,
            mfe_r=mfe_r,
            mae_r=mae_r,
            max_r=max_r,
            realized_r=realized_r,
            pnl_usd=pnl_usd,
            duration_seconds=duration,
            exit_reason=exit_reason,
            post_exit_behavior=post_exit,
        )

        return outcome_data, outcome, loss_cause, loss_cat

    def evaluate_unexecuted(
        self,
        entry: JournalEntry,
        candles: Sequence[Candle],
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> tuple[OutcomeData, TradeOutcome, LossRootCause | None, LossCategory | None]:
        risk_dist = abs(entry_price - stop_loss)
        if risk_dist <= 1e-6:
            risk_dist = max(1.0, entry.snapshot.atr_m5)

        entry_hit = False
        hit_index = -1

        for idx, c in enumerate(candles):
            bull_hit = entry.direction == Direction.BULLISH and c.low <= entry_price
            bear_hit = entry.direction == Direction.BEARISH and c.high >= entry_price
            if bull_hit or bear_hit:
                entry_hit = True
                hit_index = idx
                break

        if not entry_hit or hit_index < 0:
            outcome_data = OutcomeData(
                executed=False,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                exit_reason="Entry level never reached",
            )
            return outcome_data, TradeOutcome.UNEXECUTED, None, None

        mfe_val = 0.0
        mae_val = 0.0
        tp_hit = False
        sl_hit = False

        for c in candles[hit_index:]:
            if entry.direction == Direction.BULLISH:
                fav = c.high - entry_price
                adv = entry_price - c.low
                if fav > mfe_val:
                    mfe_val = fav
                if adv > mae_val:
                    mae_val = adv

                # Conservative evaluation: SL checked before TP
                if c.low <= stop_loss:
                    sl_hit = True
                    break
                if c.high >= take_profit:
                    tp_hit = True
                    break
            else:
                fav = entry_price - c.low
                adv = c.high - entry_price
                if fav > mfe_val:
                    mfe_val = fav
                if adv > mae_val:
                    mae_val = adv

                # Conservative evaluation: SL checked before TP
                if c.high >= stop_loss:
                    sl_hit = True
                    break
                if c.low <= take_profit:
                    tp_hit = True
                    break

        mfe_r = round(mfe_val / risk_dist, 2)
        mae_r = round(mae_val / risk_dist, 2)

        if sl_hit:
            outcome = TradeOutcome.LOSS
            realized_r = -1.0
            exit_reason = "Hypothetical SL hit"
            loss_cause, loss_cat = self.diagnose_loss(
                entry.snapshot,
                risk_dist,
                exit_reason="Hypothetical SL",
                direction=entry.direction,
            )
        elif tp_hit:
            outcome = TradeOutcome.WIN
            realized_r = round(abs(take_profit - entry_price) / risk_dist, 2)
            exit_reason = "Hypothetical TP reached"
            loss_cause = None
            loss_cat = None
        else:
            outcome = TradeOutcome.UNEXECUTED
            realized_r = 0.0
            exit_reason = "Hypothetical position expired"
            loss_cause = None
            loss_cat = None

        outcome_data = OutcomeData(
            executed=False,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            fill_price=entry_price,
            exit_price=stop_loss if sl_hit else (take_profit if tp_hit else None),
            mfe_r=mfe_r,
            mae_r=mae_r,
            max_r=max(mfe_r, realized_r),
            realized_r=realized_r,
            exit_reason=exit_reason,
        )

        return outcome_data, outcome, loss_cause, loss_cat

    def diagnose_loss(
        self,
        snapshot: SnapshotData,
        risk_dist: float,
        exit_reason: str = "",
        direction: Direction | str | None = None,
    ) -> tuple[LossRootCause, LossCategory]:
        htf_bias_str = snapshot.htf_bias.upper()
        trade_dir = (
            direction.value.upper()
            if isinstance(direction, Direction)
            else (str(direction).upper() if direction is not None else "")
        )
        if not trade_dir:
            if "BULL" in snapshot.user_decision.upper():
                trade_dir = "BULLISH"
            elif "BEAR" in snapshot.user_decision.upper():
                trade_dir = "BEARISH"

        if trade_dir:
            if "BEAR" in htf_bias_str and "BULL" in trade_dir:
                return LossRootCause.COUNTER_BIAS, LossCategory.BAD_DECISION
            if "BULL" in htf_bias_str and "BEAR" in trade_dir:
                return LossRootCause.COUNTER_BIAS, LossCategory.BAD_DECISION

        if snapshot.displacement_atr < self.min_displacement_atr:
            return LossRootCause.INSUFFICIENT_DISPLACEMENT, LossCategory.BAD_DECISION

        if (
            snapshot.news_distance_min is not None
            and abs(snapshot.news_distance_min) <= self.news_blackout_window_min
        ):
            return LossRootCause.NEWS_PROXIMITY, LossCategory.BAD_DECISION

        if (
            snapshot.atr_m5 > 0
            and (snapshot.spread_price / snapshot.atr_m5) > self.max_spread_atr_ratio
        ):
            return LossRootCause.HIGH_SPREAD, LossCategory.BAD_DECISION

        if snapshot.atr_m5 > 0 and (risk_dist / snapshot.atr_m5) < self.min_sl_atr_ratio:
            return LossRootCause.TIGHT_SL, LossCategory.BAD_DECISION

        if (
            "REJECT" in exit_reason.upper()
            or "TIMEOUT" in exit_reason.upper()
            or "EXECUTION" in exit_reason.upper()
        ):
            return LossRootCause.EXECUTION_FAILURE, LossCategory.BAD_DECISION

        if (
            "TRAILING" in exit_reason.upper()
            or "PREMATURE" in exit_reason.upper()
            or "EARLY" in exit_reason.upper()
        ):
            return LossRootCause.PREMATURE_EXIT, LossCategory.BAD_DECISION

        return LossRootCause.NORMAL_VARIANCE, LossCategory.NORMAL_VARIANCE
