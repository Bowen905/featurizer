import torch
from featurizer.functions import time_series_functions as tsf
from featurizer.interface import Functor


class Alpha001(Functor):

    def forward(self, close_ts, returns_ts):
        zeros = torch.zeros(returns_ts.size())
        downside_stds = tsf.rolling_downside_std(returns_ts, zeros, 20)
        inner = close_ts.clone()
        cond = returns_ts < 0
        inner = torch.where(cond, downside_stds, inner)
        output_tensor = tsf.rank(tsf.ts_argmax(inner ** 2, 5))
        return output_tensor


class Alpha002(Functor):

    def forward(self, open_ts, close_ts, volume_ts):  # volume
        output_tensor = -1 * tsf.rolling_corr(tsf.rank(tsf.diff(torch.log(volume_ts), 2)),
                                   tsf.rank((close_ts - open_ts) / open_ts), 6)
        return output_tensor


class Alpha003(Functor):

    def forward(self, open_ts, volume_ts):
        output_tensor = -1 * tsf.rolling_corr(tsf.rank(open_ts), tsf.rank(volume_ts), 10)
        return output_tensor


class Alpha004(Functor):

    def forward(self, low_ts):
        alpha = -1 * tsf.ts_rank(tsf.rank(low_ts), 9)
        return alpha


class Alpha006(Functor):

    def forward(self, open_ts, volume_ts):
        result = -1 * tsf.rolling_corr(open_ts, volume_ts, 10)
        return result


class Alpha007(Functor):

    def forward(self, close_ts, volume_ts):
        adv20 = tsf.rolling_mean_(volume_ts, 20)
        alpha = -1 * tsf.ts_rank(abs(tsf.diff(close_ts, 7)), 60) * torch.sign(tsf.diff(close_ts, 7))
        cond = adv20 >= volume_ts
        ones = torch.ones(close_ts.size())
        output_tensor = torch.where(cond, -1 * ones, alpha.float())  # confusing why must .float()
        return output_tensor


class Alpha008(Functor):

    def forward(self, open_ts, returns_ts):
        output_tensor = -1 * (tsf.rank(((tsf.rolling_sum_(open_ts, 5) * tsf.rolling_sum_(returns_ts, 5)) -
                                        tsf.shift((tsf.rolling_sum_(open_ts, 5) * tsf.rolling_sum_(returns_ts, 5)),
                                                  10))))
        return output_tensor


class Alpha009(Functor):

    def forward(self, close_ts):
        delta_close = tsf.diff(close_ts, 1)
        cond_1 = tsf.rolling_max(delta_close, 5) > 0
        cond_2 = tsf.rolling_max(delta_close, 5) < 0
        alpha = -1 * delta_close
        result = torch.where((cond_1 | cond_2), alpha, delta_close)
        return result


class Alpha010(Functor):

    def forward(self, close_ts):
        delta_close = tsf.diff(close_ts, 1)
        cond_1 = tsf.rolling_max(delta_close, 4) > 0
        cond_2 = tsf.rolling_max(delta_close, 4) < 0
        alpha = -1 * delta_close
        result = torch.where((cond_1 | cond_2), alpha, delta_close)
        return result


class Alpha9_10_customizable(Functor):

    def __init__(self, window):
        self.window = window

    def forward(self, close_ts):
        delta_close = tsf.diff(close_ts, 1)
        cond_1 = (tsf.rolling_max(delta_close, self.window) > 0).squeeze()
        cond_2 = (tsf.rolling_max(delta_close, self.window) < 0).squeeze()
        alpha = -1 * delta_close
        result = torch.where((cond_1 | cond_2), alpha, delta_close)
        return result


class Alpha012(Functor):

    def forward(self, close_ts, volume_ts):
        return torch.sign(tsf.diff(volume_ts, 1)) * (-1 * tsf.diff(close_ts, 1))


class Alpha013(Functor):

    def forward(self, close_ts, volume_ts):
        alpha = -1 * tsf.rank(tsf.rolling_cov(tsf.rank(close_ts), tsf.rank(volume_ts), 5))
        return alpha


class Alpha014(Functor):

    def forward(self, open_ts, volume_ts, returns_ts):
        df = tsf.rolling_cov(open_ts, volume_ts, 10)
        alpha = -1 * tsf.rank(tsf.diff(returns_ts, 3)) * df
        return alpha


class Alpha015(Functor):

    def forward(self, high_ts, volume_ts):
        df = tsf.rolling_cov(tsf.rank(high_ts), tsf.rank(volume_ts), 3)
        alpha = -1 * tsf.rolling_sum_(tsf.rank(df), 3)
        return alpha


class Alpha016(Functor):

    def forward(self, high_ts, volume_ts):
        alpha = -1 * tsf.rank(tsf.rolling_cov(tsf.rank(high_ts), tsf.rank(volume_ts), 5))
        return alpha


class Alpha017(Functor):

    def forward(self, close_ts, volume_ts):
        adv20 = tsf.rolling_mean_(volume_ts, 20)
        alpha = -1 * (tsf.rank(tsf.ts_rank(close_ts, 10)) *
                      tsf.rank(tsf.diff(tsf.diff(close_ts, 1), 1)) *
                      tsf.rank(tsf.ts_rank((volume_ts / adv20), 5)))
        return alpha


class Alpha018(Functor):

    def forward(self, open_ts, close_ts):
        ts = tsf.rolling_cov(close_ts, open_ts, 10)
        alpha = -1 * (tsf.rank((tsf.rolling_std(abs((close_ts - open_ts)), 5) + (close_ts - open_ts)) + ts))
        return alpha


class Alpha019(Functor):

    def forward(self, close_ts, returns_ts):
        alpha = ((-1 * torch.sign((close_ts - tsf.shift(close_ts, 7)) + tsf.diff(close_ts, 7))) *
                 (1 + tsf.rank(1 + tsf.rolling_sum_(returns_ts, 210))))
        return alpha


class Alpha020(Functor):

    def forward(self, open_ts, high_ts, low_ts, close_ts):
        alpha = -1 * (tsf.rank(open_ts - tsf.shift(high_ts, 1)) *
                      tsf.rank(open_ts - tsf.shift(close_ts, 1)) *
                      tsf.rank(open_ts - tsf.shift(high_ts, 1)))
        return alpha


class Alpha021(Functor):

    def forward(self, close_ts, volume_ts):
        cond_1 = (tsf.rolling_mean_(close_ts, 8) + tsf.rolling_std(close_ts, 8) < tsf.rolling_mean_(close_ts,
                                                                                                    2)).squeeze()
        cond_2 = (tsf.rolling_mean_(volume_ts, 20) / volume_ts < 1).squeeze()
        cond_3 = (tsf.rolling_mean_(close_ts, 8) - tsf.rolling_std(close_ts, 8) < tsf.rolling_mean_(close_ts,
                                                                                       2)).squeeze()
        ones = torch.ones(close_ts.size()).squeeze()
        result = torch.where((cond_1 | (cond_2 & cond_3)), -1 * ones, ones)
        return result


class Alpha022(Functor):

    def forward(self, high_ts, close_ts, volume_ts):
        ts = tsf.rolling_cov(high_ts, volume_ts, 5)
        alpha = -1 * tsf.diff(ts, 5) * tsf.rank(tsf.rolling_std(close_ts, 20))
        return alpha


class Alpha023(Functor):

    def forward(self, high_ts):
        cond = (tsf.rolling_mean_(high_ts, 20) < high_ts).squeeze()
        zeros = torch.zeros_like(high_ts).squeeze()
        result = torch.where(cond, -1 * tsf.diff(high_ts, 2).squeeze(), zeros)
        return result


class Alpha024(Functor):

    def forward(self, close_ts):
        cond = (tsf.diff(tsf.rolling_mean_(close_ts, 100), 100) / tsf.shift(close_ts, 100) <= 0.05).squeeze()
        alpha = -1 * tsf.diff(close_ts, 3).squeeze()
        result = torch.where(cond, -1 * (close_ts - tsf.rolling_min(close_ts, 100)).squeeze(), alpha)
        return result


class Alpha024_customizable(Functor):

    def __init__(self, window):
        self.window = window

    def forward(self, close_ts):
        cond = (tsf.diff(tsf.rolling_mean_(close_ts, self.window), self.window) / tsf.shift(close_ts,
                                                                                            self.window) <= 0.05).squeeze()
        alpha = -1 * tsf.diff(close_ts, 3).squeeze()
        result = torch.where(cond, -1 * (close_ts - tsf.rolling_min(close_ts, self.window)).squeeze(), alpha)
        return result


class Alpha026(Functor):

    def forward(self, high_ts, volume_ts):
        ts = tsf.rolling_cov(tsf.ts_rank(volume_ts, 5), tsf.ts_rank(high_ts, 5), 5)
        alpha = -1 * tsf.rolling_max(ts, 3)
        return alpha


class Alpha028(Functor):

    def forward(self, high_ts, low_ts, close_ts, volume_ts):
        adv20 = tsf.rolling_mean_(volume_ts, 20)
        df = tsf.rolling_cov(adv20, low_ts, 5)
        alpha = tsf.rolling_scale(((df + ((high_ts + low_ts) / 2)) - close_ts))
        return alpha


class Alpha029(Functor):

    def forward(self, close_ts, returns_ts):
        return tsf.rolling_min(tsf.rank(tsf.rank(tsf.rolling_scale(
            torch.log(tsf.rolling_sum_(tsf.rank(tsf.rank(-1 * tsf.rank(tsf.diff((close_ts - 1), 5)))), 2))))),
            5) + tsf.ts_rank(tsf.shift((-1 * returns_ts), 6), 5)


class Alpha030(Functor):

    def forward(self, close_ts, volume_ts):
        delta_close = tsf.diff(close_ts, 1)
        inner = torch.sign(delta_close) + torch.sign(tsf.shift(delta_close, 1)) + torch.sign(tsf.shift(delta_close, 2))
        alpha = ((1.0 - tsf.rank(inner)) * tsf.rolling_sum_(volume_ts, 5)) / tsf.rolling_sum_(volume_ts, 20)
        return alpha


# class Alpha031(Functor):  # decay_linear
#
#    def forward(self, low_ts, close_ts, volume_ts):
#         adv20 = tsf.rolling_mean(volume_ts, 20)
#         ts = tsf.rolling_corr(adv20, low_ts, 12)
#         output_tensor=(tsf.rank(
#             tsf.rank(tsf.rank(tsf.decay_linear(-1 * func.rank(func.rank(func.delta(close_np, 10))), 10)))) +
#                  func.rank(-1 * func.delta(close_np, 3)) ) + np.sign(func.scale(df))
#         return output_tensor


class Alpha033(Functor):

    def forward(self, open_ts, close_ts):
        alpha = tsf.rank(-1 + (open_ts / close_ts))
        return alpha


class Alpha034(Functor):

    def forward(self, close_ts, returns_ts):
        inner = tsf.rolling_std(returns_ts, 2) / tsf.rolling_std(returns_ts, 5)
        return tsf.rank(2 - tsf.rank(inner) - tsf.rank(tsf.diff(close_ts, 1)))  # .values


class Alpha035(Functor):

    def forward(self, high_ts, low_ts, close_ts, volume_ts, returns_ts):
        return (tsf.ts_rank(volume_ts, 32) * (1 - tsf.ts_rank(close_ts + high_ts - low_ts, 16))) * (
                1 - tsf.ts_rank(returns_ts, 32))


class Alpha037(Functor):

    def forward(self, open_ts, close_ts):
        alpha = tsf.rank(tsf.rolling_corr(tsf.shift(open_ts - close_ts, 1), close_ts, 200)) + tsf.rank(
            open_ts - close_ts)
        return alpha


class Alpha038(Functor):

    def forward(self, open_ts, close_ts):
        inner = close_ts / open_ts
        alpha = -1 * tsf.rank(tsf.ts_rank(open_ts, 10)) * tsf.rank(inner)
        return alpha


# class Alpha039(Functor):  # decay_linear
#
#    def forward(self, close_ts, volume_ts, returns_ts):
#         adv20 = func.sma(volume_np, 20)
#         return ((-1 * func.rank(func.delta(close_np, 7) * (1 - func.rank(func.decay_linear(volume_np / adv20, 9))))) *
#                 (1 + func.rank(func.ts_sum(returns_np, 250))))

class Alpha040(Functor):

    def forward(self, high_ts, volume_ts):
        alpha = -1 * tsf.rank(tsf.rolling_std(high_ts, 10)) * tsf.rolling_corr(high_ts, volume_ts, 10)
        return alpha  #


class Alpha043(Functor):

    def forward(self, close_ts, volume_ts):
        adv20 = tsf.rolling_mean_(volume_ts, 20)
        alpha = tsf.ts_rank(volume_ts / adv20, 20) * tsf.ts_rank((-1 * tsf.diff(close_ts, 7)), 8)
        return alpha


class Alpha044(Functor):

    def forward(self, high_ts, volume_ts):
        output_tensor = tsf.rolling_corr(high_ts, tsf.rank(volume_ts), 5)
        return -1 * output_tensor


class Alpha045(Functor):

    def forward(self, close_ts, volume_ts):
        ts = tsf.rolling_corr(close_ts, volume_ts, 2)
        return -1 * (tsf.rank(tsf.rolling_mean_(tsf.shift(close_ts, 5), 20)) * ts *
                     tsf.rank(
                         tsf.rolling_corr(tsf.rolling_sum_(close_ts, 5), tsf.rolling_sum_(close_ts, 20), 2)))  # .values


class Alpha046(Functor):

    def forward(self, close_ts):
        inner = ((tsf.shift(close_ts, 20) - tsf.shift(close_ts, 10)) / 10).squeeze() - (
                    (tsf.shift(close_ts, 10) - close_ts) / 10).squeeze()
        cond_1 = inner < 0
        cond_2 = inner > 0.25
        alpha = (-1 * tsf.diff(close_ts).squeeze())
        ones = torch.ones_like(close_ts).squeeze()
        result = torch.where((cond_1 | cond_2), -1 * ones, alpha)
        return result


class Alpha049(Functor):

    def forward(self, close_ts):
        inner = (((tsf.shift(close_ts, 20) - tsf.shift(close_ts, 10)) / 10).squeeze() - (
                (tsf.shift(close_ts, 10) - close_ts) / 10).squeeze())
        alpha = (-1 * tsf.diff(close_ts)).squeeze()
        cond = (inner < -0.1)
        ones = torch.ones_like(close_ts).squeeze()
        result = torch.where(cond, ones, alpha)
        return result


class Alpha051(Functor):

    def forward(self, close_ts):
        inner = (((tsf.shift(close_ts, 20) - tsf.shift(close_ts, 10)) / 10) - (
                (tsf.shift(close_ts, 10) - close_ts) / 10)).squeeze()
        alpha = (-1 * tsf.diff(close_ts)).squeeze()
        ones = torch.ones_like(close_ts).squeeze()
        cond = (inner < -0.05).squeeze()
        result = torch.where(cond, ones, alpha)
        return result


class Alpha052(Functor):

    def forward(self, low_ts, volume_ts, returns_ts):
        return ((-1 * tsf.diff(tsf.rolling_min(low_ts, 5), 5)) *
                tsf.rank(((tsf.rolling_sum_(returns_ts, 60) - tsf.rolling_sum_(returns_ts, 20)) / 55))) * tsf.ts_rank(
            volume_ts, 5)


class Alpha053(Functor):

    def forward(self, high_ts, low_ts, close_ts):
        inner = (close_ts - low_ts).squeeze()
        constant = 0.0001 * torch.ones_like(close_ts).squeeze()
        cond = inner == 0
        inner = torch.where(cond, constant, inner)
        result = -1 * tsf.diff((((close_ts - low_ts) - (high_ts - close_ts)).squeeze() / inner), 9)
        return result


class Alpha054(Functor):

    def forward(self, open_ts, high_ts, low_ts, close_ts):
        inner = (low_ts - high_ts).squeeze()
        constant = 0.0001 * torch.ones_like(close_ts).squeeze()
        cond = inner == 0
        inner = torch.where(cond, constant, inner)
        result = -1 * ((low_ts - close_ts) * (open_ts ** 5)).squeeze() / (inner * (close_ts ** 5).squeeze())
        return result


class Alpha055(Functor):

    def forward(self, high_ts, low_ts, close_ts, volume_ts):
        inner = tsf.rolling_max(high_ts, 12) - tsf.rolling_min(low_ts, 12)
        cond = inner == 0
        constant = 0.0001 * torch.ones_like(high_ts)
        divisor = torch.where(cond, constant, inner)
        inner_ = (close_ts - tsf.rolling_min(low_ts, 12)) / (divisor)
        ts = tsf.rolling_corr(tsf.rank(inner_), tsf.rank(volume_ts), 6)
        output_tensor = -1 * ts
        return output_tensor
