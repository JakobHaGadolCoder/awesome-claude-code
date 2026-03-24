"""
Options Greeks Calculator
Implements Black-Scholes pricing with full Greeks:
Delta, Gamma, Theta, Vega, Rho, Vanna, Charm.
Also provides IV surface fitting and strategy P&L calculations.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from options_trader.core.models import OptionContract, OptionType

logger = logging.getLogger(__name__)

N = norm.cdf
n = norm.pdf


@dataclass
class GreeksResult:
    price: float
    delta: float
    gamma: float
    theta: float      # per calendar day
    vega: float       # per 1% IV move
    rho: float        # per 1% interest rate move
    vanna: float      # dDelta/dVol (second order)
    charm: float      # dDelta/dTime (second order)
    implied_volatility: float
    intrinsic_value: float
    time_value: float


class BlackScholes:
    """
    Black-Scholes option pricing and full Greeks computation.
    """

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return float("inf")
        return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

    @staticmethod
    def d2(d1_val: float, sigma: float, T: float) -> float:
        if T <= 0:
            return float("inf")
        return d1_val - sigma * math.sqrt(T)

    @classmethod
    def price(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
    ) -> float:
        """Theoretical option price."""
        if T <= 0:
            # Expired
            if option_type == OptionType.CALL:
                return max(0.0, S - K)
            return max(0.0, K - S)

        d1 = cls.d1(S, K, T, r, sigma)
        d2 = cls.d2(d1, sigma, T)

        if option_type == OptionType.CALL:
            return S * N(d1) - K * math.exp(-r * T) * N(d2)
        return K * math.exp(-r * T) * N(-d2) - S * N(-d1)

    @classmethod
    def greeks(
        cls,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
    ) -> GreeksResult:
        """Compute full option price and all Greeks."""
        theoretical_price = cls.price(S, K, T, r, sigma, option_type)

        if T <= 0 or sigma <= 0:
            intrinsic = max(0.0, S - K) if option_type == OptionType.CALL else max(0.0, K - S)
            return GreeksResult(
                price=theoretical_price,
                delta=1.0 if (option_type == OptionType.CALL and S > K) else 0.0,
                gamma=0.0,
                theta=0.0,
                vega=0.0,
                rho=0.0,
                vanna=0.0,
                charm=0.0,
                implied_volatility=sigma,
                intrinsic_value=intrinsic,
                time_value=0.0,
            )

        d1_val = cls.d1(S, K, T, r, sigma)
        d2_val = cls.d2(d1_val, sigma, T)
        sqrt_T = math.sqrt(T)
        exp_rT = math.exp(-r * T)

        # Delta
        if option_type == OptionType.CALL:
            delta = N(d1_val)
        else:
            delta = N(d1_val) - 1.0

        # Gamma (same for calls and puts)
        gamma = n(d1_val) / (S * sigma * sqrt_T)

        # Theta (per calendar day, divided by 365)
        if option_type == OptionType.CALL:
            theta = (
                -(S * n(d1_val) * sigma) / (2 * sqrt_T)
                - r * K * exp_rT * N(d2_val)
            ) / 365.0
        else:
            theta = (
                -(S * n(d1_val) * sigma) / (2 * sqrt_T)
                + r * K * exp_rT * N(-d2_val)
            ) / 365.0

        # Vega (per 1% change in IV)
        vega = S * n(d1_val) * sqrt_T / 100.0

        # Rho (per 1% change in interest rate)
        if option_type == OptionType.CALL:
            rho = K * T * exp_rT * N(d2_val) / 100.0
        else:
            rho = -K * T * exp_rT * N(-d2_val) / 100.0

        # Vanna = dDelta/dVol = dVega/dS
        vanna = -n(d1_val) * d2_val / sigma

        # Charm = dDelta/dTime (rate of delta decay)
        if option_type == OptionType.CALL:
            charm = -n(d1_val) * (
                2 * r * T - d2_val * sigma * sqrt_T
            ) / (2 * T * sigma * sqrt_T)
        else:
            charm = -n(d1_val) * (
                2 * r * T - d2_val * sigma * sqrt_T
            ) / (2 * T * sigma * sqrt_T)

        intrinsic = max(0.0, S - K) if option_type == OptionType.CALL else max(0.0, K - S)
        time_value = max(0.0, theoretical_price - intrinsic)

        return GreeksResult(
            price=theoretical_price,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
            vanna=vanna,
            charm=charm,
            implied_volatility=sigma,
            intrinsic_value=intrinsic,
            time_value=time_value,
        )

    @staticmethod
    def implied_volatility(
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: OptionType,
        tol: float = 1e-6,
        max_iter: int = 200,
    ) -> Optional[float]:
        """Compute IV using Brent's method."""
        if T <= 0 or market_price <= 0:
            return None

        intrinsic = max(0.0, S - K) if option_type == OptionType.CALL else max(0.0, K - S)
        if market_price < intrinsic - tol:
            return None

        def objective(sigma: float) -> float:
            return BlackScholes.price(S, K, T, r, sigma, option_type) - market_price

        try:
            iv = brentq(objective, 1e-6, 10.0, xtol=tol, maxiter=max_iter)
            return float(iv)
        except (ValueError, RuntimeError):
            return None


class OptionsGreeksCalculator:
    """
    High-level wrapper that:
    - Computes Greeks for entire options chains
    - Calculates IV surface (skew / term structure)
    - Evaluates strategy-level P&L (spreads, straddles, etc.)
    """

    def __init__(self, risk_free_rate: float = 0.05):
        self.r = risk_free_rate
        self.bs = BlackScholes()

    # ------------------------------------------------------------------
    # Greeks for a single contract
    # ------------------------------------------------------------------

    def compute_greeks(
        self, contract: OptionContract, risk_free_rate: Optional[float] = None
    ) -> GreeksResult:
        r = risk_free_rate or self.r
        S = contract.underlying_price
        K = contract.strike
        T = max(0.0, (contract.expiration - datetime.utcnow()).total_seconds() / (365.25 * 86400))
        sigma = contract.implied_volatility if contract.implied_volatility > 0 else 0.20

        return self.bs.greeks(S, K, T, r, sigma, contract.option_type)

    def price_contract(
        self, contract: OptionContract, risk_free_rate: Optional[float] = None
    ) -> float:
        r = risk_free_rate or self.r
        S = contract.underlying_price
        K = contract.strike
        T = max(0.0, (contract.expiration - datetime.utcnow()).total_seconds() / (365.25 * 86400))
        sigma = contract.implied_volatility if contract.implied_volatility > 0 else 0.20
        return self.bs.price(S, K, T, r, sigma, contract.option_type)

    def solve_iv(
        self, contract: OptionContract, market_price: float
    ) -> Optional[float]:
        S = contract.underlying_price
        K = contract.strike
        T = max(0.0, (contract.expiration - datetime.utcnow()).total_seconds() / (365.25 * 86400))
        return self.bs.implied_volatility(market_price, S, K, T, self.r, contract.option_type)

    # ------------------------------------------------------------------
    # Chain-level analysis
    # ------------------------------------------------------------------

    def analyze_chain(
        self, contracts: List[OptionContract]
    ) -> Dict[str, GreeksResult]:
        """Compute Greeks for every contract. Returns {contract_key: GreeksResult}."""
        results = {}
        for c in contracts:
            key = f"{c.symbol}_{c.option_type.value}_{c.strike}_{c.expiration.date()}"
            try:
                results[key] = self.compute_greeks(c)
            except Exception as exc:
                logger.debug("Greeks error for %s: %s", key, exc)
        return results

    def net_delta_exposure(
        self, contracts: List[OptionContract], position_sizes: Dict[str, int]
    ) -> float:
        """Portfolio net delta (sum of delta * contracts * 100 shares)."""
        net = 0.0
        for c in contracts:
            key = f"{c.symbol}_{c.option_type.value}_{c.strike}_{c.expiration.date()}"
            size = position_sizes.get(key, 0)
            g = self.compute_greeks(c)
            net += g.delta * size * 100
        return net

    # ------------------------------------------------------------------
    # Strategy P&L profiles
    # ------------------------------------------------------------------

    def long_call_pnl(
        self, K: float, premium_paid: float, price_range: np.ndarray
    ) -> np.ndarray:
        return np.maximum(price_range - K, 0) - premium_paid

    def long_put_pnl(
        self, K: float, premium_paid: float, price_range: np.ndarray
    ) -> np.ndarray:
        return np.maximum(K - price_range, 0) - premium_paid

    def bull_call_spread_pnl(
        self,
        K_low: float,
        K_high: float,
        net_debit: float,
        price_range: np.ndarray,
    ) -> np.ndarray:
        long_call = np.maximum(price_range - K_low, 0)
        short_call = np.maximum(price_range - K_high, 0)
        return long_call - short_call - net_debit

    def bear_put_spread_pnl(
        self,
        K_high: float,
        K_low: float,
        net_debit: float,
        price_range: np.ndarray,
    ) -> np.ndarray:
        long_put = np.maximum(K_high - price_range, 0)
        short_put = np.maximum(K_low - price_range, 0)
        return long_put - short_put - net_debit

    def straddle_pnl(
        self,
        K: float,
        total_premium: float,
        price_range: np.ndarray,
    ) -> np.ndarray:
        call_payoff = np.maximum(price_range - K, 0)
        put_payoff = np.maximum(K - price_range, 0)
        return call_payoff + put_payoff - total_premium

    def iron_condor_pnl(
        self,
        K_put_low: float,
        K_put_high: float,
        K_call_low: float,
        K_call_high: float,
        net_credit: float,
        price_range: np.ndarray,
    ) -> np.ndarray:
        short_put = -np.maximum(K_put_high - price_range, 0)
        long_put = np.maximum(K_put_low - price_range, 0)
        short_call = -np.maximum(price_range - K_call_low, 0)
        long_call = np.maximum(price_range - K_call_high, 0)
        return short_put + long_put + short_call + long_call + net_credit

    # ------------------------------------------------------------------
    # Expected move calculation
    # ------------------------------------------------------------------

    def expected_move(
        self,
        underlying_price: float,
        iv: float,
        dte: int,
        confidence: float = 0.68,
    ) -> Tuple[float, float]:
        """
        1-std expected move range for given DTE and IV.
        Returns (lower_bound, upper_bound).
        confidence=0.68 → 1-sigma, 0.95 → 2-sigma
        """
        T = dte / 365.0
        z = norm.ppf((1 + confidence) / 2)
        move = underlying_price * iv * math.sqrt(T) * z
        return (underlying_price - move, underlying_price + move)

    def breakeven_prices(
        self,
        strategy: str,
        strikes: List[float],
        net_cost: float,
    ) -> List[float]:
        """Calculate breakeven price(s) for common strategies."""
        if strategy == "long_call":
            return [strikes[0] + net_cost]
        elif strategy == "long_put":
            return [strikes[0] - net_cost]
        elif strategy == "straddle":
            K = strikes[0]
            return [K - net_cost, K + net_cost]
        elif strategy == "bull_call_spread":
            return [strikes[0] + net_cost]
        elif strategy == "bear_put_spread":
            return [strikes[0] - net_cost]
        return []
