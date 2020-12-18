import torch

from featurizer.functions import time_series_functions as tsf
from featurizer.interface import Functor


class Alpha001(Functor):

    def forward(self, volume_ts, close_ts, open_ts):
        factor1 = tsf.rank(tsf.diff(torch.log(volume_ts), 1))
        factor2 = tsf.rank((close_ts - open_ts) / open_ts)
        output_tensor = tsf.rolling_corr(factor1, factor2, 6)
        return output_tensor


class Alpha002(Functor):

    def forward(self, close_ts, high_ts, low_ts):
        output_tensor = (-1 * tsf.diff((((close_ts - low_ts) - (high_ts - close_ts)) / (high_ts - low_ts)), 1))
        return output_tensor


class Alpha003(Functor):

    def forward(self, close_ts, low_ts, high_ts):
        cond1 = close_ts == tsf.shift(close_ts, 1)
        cond2 = close_ts > tsf.shift(close_ts, 1)
        zeros = torch.zeros_like(close_ts)
        consequance1 = close_ts - torch.min(low_ts, tsf.shift(close_ts, 1))
        consequance2 = close_ts - torch.max(high_ts, tsf.shift(close_ts, 1))
        inner = torch.where(cond2, consequance1, consequance2)
        output_tensor = torch.where(cond1, zeros, inner)
        return output_tensor


class Alpha004(Functor):

    def forward(self, close_ts, volume_ts):
        cond1 = (tsf.rolling_mean_(close_ts, 8) + tsf.rolling_std(close_ts, 8)) < tsf.rolling_mean_(close_ts, 2)
        cond2 = tsf.rolling_mean_(close_ts, 2) < (tsf.rolling_mean_(close_ts, 8) - tsf.rolling_std(close_ts, 8))
        cond3 = (volume_ts / tsf.rolling_mean_(volume_ts, 20)) >= 1
        ones = torch.ones(close_ts.size())
        output_tensor = torch.where((cond1 | ~(cond1 & cond2 & cond3)), -1 * ones, ones)
        return output_tensor


class Alpha005(Functor):

    def forward(self, volume_ts, high_ts):
        output_tensor = -1 * tsf.rolling_max(tsf.rolling_corr(tsf.ts_rank(volume_ts, 5), tsf.ts_rank(high_ts, 5), 5), 3)
        return output_tensor


# class Alpha006(Functor):
#
#     def forward(self, high_ts, open_ts):
#         output_tensor = -1 * tsf.rank(torch.sign(tsf.diff((((open_ts * 0.85) + (high_ts * 0.15))), 4)))
#         return output_tensor


# class Alpha007(Functor): # what max?
#
#     def forward(self, vwap_ts, close_ts, volume_ts):
#         output_tensor = ((tsf.rank(tsf.rolling_max((vwap_ts - close_ts), 3)) + tsf.rank(
#             ((vwap_ts - close_ts), 3))) * tsf.rank(tsf.diff(volume_ts, 3)))
#         return output_tensor


class Alpha008(Functor):

    def forward(self, vwap_ts, high_ts, low_ts):
        output_tensor = tsf.rank(tsf.diff(((((high_ts + low_ts) / 2) * 0.2) + (vwap_ts * 0.8)), 4) * -1)
        return output_tensor


# class Alpha009(Functor):
#
#     def forward(self, high_ts, low_ts, volume_ts):
#         output_tensor = tsf.rolling_mean_(
#             ((high_ts + low_ts) / 2 - (tsf.shift(high_ts, 1) + tsf.shift(low_ts, 1)) / 2) * (
#                     high_ts - low_ts) / volume_ts, 7, 2)
#         return output_tensor


# class Alpha010(Functor):  #  BUG in formula?
#
#     def forward(self, high_ts,return_ts,close_ts):
#         cond1 = return_ts<0
#         inner = torch.where(cond1,tsf.rolling_std(return_ts,20),close_ts)**2
#         output_tensor = tsf.rank(tsf.rolling_max(inner,5))
#         return output_tensor


class Alpha011(Functor):

    def forward(self, high_ts, low_ts, close_ts, volume_ts):
        output_tensor = tsf.rolling_sum_(((close_ts - low_ts) - (high_ts - close_ts)) / (high_ts - low_ts) * volume_ts,
                                         6)
        return output_tensor


class Alpha012(Functor):

    def forward(self, open_ts, close_ts, vwap_ts):
        output_tensor = (tsf.rank((open_ts - (tsf.rolling_sum_(vwap_ts, 10) / 10)))) * (
                -1 * (tsf.rank(abs((close_ts - vwap_ts)))))
        return output_tensor


class Alpha013(Functor):

    def forward(self, high_ts, low_ts, vwap_ts):
        output_tensor = (((high_ts * low_ts) ** 0.5) - vwap_ts)
        return output_tensor


class Alpha014(Functor):

    def forward(self, close_ts):
        output_tensor = close_ts - tsf.shift(close_ts, 5)
        return output_tensor


class Alpha015(Functor):

    def forward(self, close_ts, open_ts):
        output_tensor = open_ts / tsf.shift(close_ts, 1) - 1
        return output_tensor


class Alpha016(Functor):

    def forward(self, vwap_ts, volume_ts):
        output_tensor = -1 * tsf.rolling_max(tsf.rank(tsf.rolling_corr(tsf.rank(volume_ts), tsf.rank(vwap_ts), 5)), 5)
        return output_tensor


# class Alpha017(Functor): # what max?
#
#     def forward(self,vwap_ts,close_ts):
#         output_tensor = tsf.rank((vwap_ts - tsf.rolling_max(vwap_ts, 15)))^tsf.diff(close_ts, 5)
#         return output_tensor


class Alpha018(Functor):

    def forward(self, close_ts):
        output_tensor = close_ts / tsf.shift(close_ts, 5)
        return output_tensor


class Alpha019(Functor):

    def forward(self, close_ts):
        cond1 = close_ts < tsf.shift(close_ts, 5)
        cond2 = close_ts == tsf.shift(close_ts, 5)
        inner = (close_ts - tsf.shift(close_ts, 5)) / tsf.shift(close_ts, 5)
        zeros = torch.zeros_like(close_ts)
        output_tensor = torch.where(cond1, inner, (close_ts - tsf.shift(close_ts, 5)) / close_ts)
        output_tensor = torch.where((~cond1 & cond2), zeros, output_tensor)
        return output_tensor


class Alpha020(Functor):

    def forward(self, close_ts):
        output_tensor = (close_ts - tsf.shift(close_ts, 6)) / tsf.shift(close_ts, 6) * 100
        return output_tensor


# class Alpha021(Functor):  # formula BUG
#
#     def forward(self, close_ts):
#         output_tensor = REGBETA(tsf.rolling_mean_(close_ts,6),SEQUENCE(6))
#         return output_tensor


# class Alpha025(Functor): # decay_linear
#
#     def forward(self):
#         ((-1 * tsf.rank((tsf.shift(CLOSE, 7) * (1 - tsf.rank(DECAYLINEAR((VOLUME / MEAN(VOLUME, 20)), 9)))))) * (
#                     1 + RANK(SUM(RET, 250))))

class Alpha026(Functor):

    def forward(self, close_ts, vwap_ts):
        output_tensor = (((tsf.rolling_sum_(close_ts, 7) / 7) - close_ts) + (
            (tsf.rolling_corr(vwap_ts, tsf.shift(close_ts, 5), 230))))
        return output_tensor


class Alpha029(Functor):

    def forward(self, close_ts, volume_ts):
        output_tensor = (close_ts - tsf.shift(close_ts, 6)) / tsf.shift(close_ts, 6) * volume_ts
        return output_tensor


class Alpha031(Functor):

    def forward(self, close_ts):
        output_tensor = (close_ts - tsf.rolling_mean_(close_ts, 12)) / tsf.rolling_mean(close_ts, 12) * 100
        return output_tensor


class Alpha032(Functor):

    def forward(self, high_ts, volume_ts):
        output_tensor = (
                -1 * tsf.rolling_sum_(tsf.rank(tsf.rolling_corr(tsf.rank(high_ts), tsf.rank(volume_ts), 3)), 3))
        return output_tensor


class Alpha033(Functor):

    def forward(self, low_ts, return_ts, volume_ts):
        output_tensor = ((((-1 * tsf.rolling_min(low_ts, 5)) + tsf.shift(tsf.rolling_min(low_ts, 5), 5)) * tsf.rank(
            ((tsf.rolling_sum_(return_ts, 240) - tsf.rolling_sum_(return_ts, 20)) / 220))) * tsf.rolling_min(volume_ts,
                                                                                                             5))
        return output_tensor


class Alpha034(Functor):
    def forward(self, close_ts):
        output_tensor = tsf.rolling_mean(close_ts, 12) / close_ts
        return output_tensor


# class Alpha036(Functor): #formula bug
#
#     def forward(self, volume_ts, vwap_ts):
#         output_tensor = tsf.rank(tsf.rolling_sum_(tsf.rolling_corr(tsf.rank(volume_ts), tsf.rank(vwap_ts)), 6), 2)
#         return output_tensor


class Alpha037(Functor):

    def forward(self, open_ts, return_ts):
        output_tensor = (-1 * tsf.rank(((tsf.rolling_sum_(open_ts, 5) * tsf.rolling_sum_(return_ts, 5)) - tsf.shift(
            (tsf.rolling_sum_(open_ts, 5) * tsf.rolling_sum_(return_ts, 5)), 10))))
        return output_tensor


class Alpha038(Functor):

    def forward(self, high_ts):
        cond = tsf.rolling_mean(high_ts, 20) < high_ts
        zeros = torch.zeros_like(high_ts)
        output_tensor = torch.where(cond, (-1 * tsf.diff(high_ts, 2)), zeros)
        return output_tensor


class Alpha040(Functor):

    def forward(self, close_ts, volume_ts):
        cond1 = close_ts > tsf.shift(close_ts, 1)
        zeros = torch.zeros_like(close_ts)
        inner1 = torch.where(cond1, volume_ts, zeros)
        cond2 = close_ts <= tsf.shift(close_ts, 1)
        inner2 = tsf.rolling_sum_(torch.where(cond2, volume_ts, zeros), 20)
        output_tensor = inner1 / inner2
        return output_tensor


class Alpha041(Functor):

    def forward(self, vwap_ts):
        output_tensor = (tsf.rank(tsf.rolling_max(tsf.diff((vwap_ts), 3), 5)) * -1)
        return output_tensor


class Alpha042(Functor):

    def forward(self, high_ts, volume_ts):
        output_tensor = ((-1 * tsf.rank(tsf.rolling_std(high_ts, 10))) * tsf.rolling_corr(high_ts, volume_ts, 10))
        return output_tensor


class Alpha043(Functor):

    def forward(self, close_ts, volume_ts):
        sign = torch.sign(tsf.diff(close_ts, 1))
        output_tensor = tsf.rolling_sum_(sign * volume_ts,6)
        return output_tensor


class Alpha045(Functor):

    def forward(self, close_ts, open_ts, vwap_ts, volume_ts):
        output_tensor = (tsf.rank(tsf.diff(((close_ts * 0.6 + open_ts * 0.4)), 1)) * tsf.rank(
            tsf.rolling_corr(vwap_ts, tsf.rolling_mean_(volume_ts, 150), 15)))
        return output_tensor


class Alpha046(Functor):

    def forward(self, close_ts):
        output_tensor = (tsf.rolling_mean_(close_ts, 3) + tsf.rolling_mean_(close_ts, 6) + tsf.rolling_mean_(close_ts,
                                                                                                             12) + tsf.rolling_mean_(
            close_ts, 24)) / (4 * close_ts)
        return output_tensor


class Alpha048(Functor):

    def forward(self, close_ts, volume_ts):
        output_tensor = (-1 * ((tsf.rank(((torch.sign((close_ts - tsf.shift(close_ts, 1))) + torch.sign(
            (tsf.shift(close_ts, 1) - tsf.shift(close_ts, 2)))) + torch.sign(
            (tsf.shift(close_ts, 2) - tsf.shift(close_ts, 3)))))) * tsf.rolling_sum_(volume_ts,
                                                                                     5)) / tsf.rolling_sum_(volume_ts,
                                                                                                            20))
        return output_tensor


class Alpha049(Functor):

    def forward(self, high_ts, low_ts):
        output_tensor = torch.max(tsf.diff(high_ts, 1), tsf.diff(low_ts, 1))
        return output_tensor


# class Alpha050(Functor):  #formula bug
#
#     def forward(self,high_ts,low_ts):
#         zeros=torch.zeros(high_ts.size())
#         cond1=(high_ts+low_ts)<=tsf.shift(high_ts,1)+tsf.shift(low_ts,1)
#         inner1=tsf.rolling_sum_(torch.where(cond1,zeros,torch.max(torch.abs(high_ts-tsf.shift(high_ts,1)),torch.abs(low_ts-tsf.shift(low_ts,1)))),12)
#         inner2=tsf.rolling_sum_(torch.where(cond1,zeros,torch.max(torch.abs(high_ts-tsf.shift(high_ts,1)),torch.abs(low_ts-tsf.shift(low_ts,1)))),12)
#         inner2=(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(L OW-DELAY(LOW,1)))),12)
#         inner3=SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HI GH,1)),ABS(LOW-DELAY(LOW,1)))),12))
#         inner4=+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELA Y(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))
#         output_tensor=inner1/inner2+inner3-inner4
#         output_tensor=torch.max(tsf.diff(high_ts,1),tsf.diff(low_ts,1))
#         return output_tensor


# class Alpha052(Functor):  #formula bug:what is L
#
#     def forward(self,high_ts,low_ts,close_ts):
#         output_tensor=tsf.rolling_sum_(MAX(0,high_ts-tsf.shift((high_ts+low_ts+close_ts)/3,1)),26)/tsf.rolling_sum_(MAX(0,tsf.shift((high_ts+low_ts+close_ts)/3,1)-L),26)*100
#         return output_tensor


# class Alpha054(Functor): #formula BUG
#
#     def forward(self, close_ts, open_ts):
#         output_tensor = (-1 * tsf.rank(
#             (tsf.rolling_std(torch.abs(close_ts - open_ts)) + (close_ts - open_ts)) + tsf.rolling_corr(close_ts,
#                                                                                                        open_ts, 10)))
#         return output_tensor


class Alpha055(Functor):

    def forward(self, close_ts, open_ts, high_ts, low_ts):
        inner1 = tsf.rolling_sum_(16 * (
                close_ts - tsf.shift(close_ts, 1) + (close_ts - open_ts) / 2 + tsf.shift(close_ts, 1) - tsf.shift(
            open_ts, 1)))
        cond1 = (torch.abs(high_ts - tsf.shift(close_ts, 1)) > torch.abs(low_ts - tsf.shift(close_ts, 1))) & (
                torch.abs(high_ts - tsf.shift(close_ts, 1)) > torch.abs(high_ts - tsf.shift(low_ts, 1)))
        sequence1 = torch.abs(high_ts - tsf.shift(close_ts, 1)) + torch.abs(
            low_ts - tsf.shift(close_ts, 1)) / 2 + torch.abs(tsf.shift(close_ts, 1) - tsf.shift(open_ts, 1)) / 4
        cond2 = (torch.abs(low_ts - tsf.shift(close_ts, 1)) > torch.abs(high_ts - tsf.shift(low_ts, 1))) & (
                torch.abs(low_ts - tsf.shift(close_ts, 1)) > torch.abs(high_ts - tsf.shift(close_ts, 1)))
        sequence2 = torch.abs(high_ts - tsf.shift(low_ts, 1)) + torch.abs(
            tsf.shift(close_ts, 1) - tsf.shift(open_ts, 1)) / 4
        inner2 = torch.where((~cond1 & ~cond2), sequence2, sequence1)
        output_tensor = inner1 / inner2
        return output_tensor


# class Alpha056(Functor):
#
#     def forward(self, open_ts, high_ts, low_ts, volume_ts):
#         output_tensor = (tsf.rank((open_ts - tsf.rolling_min(open_ts, 12))) < tsf.rank(tsf.rank(
#             tsf.rolling_corr(tsf.rolling_sum_(((high_ts + low_ts) / 2), 19),
#                              tsf.rolling_sum_(tsf.rolling_mean_(volume_ts, 40), 19), 13)) ** 5))
#         return output_tensor


class Alpha059(Functor):

    def forward(self, close_ts, low_ts, high_ts):
        cond1 = tsf.diff(close_ts, 1) > 0
        zeros = torch.zeros_like(close_ts)
        inner = torch.where(cond1, torch.min(low_ts, tsf.shift(close_ts, 1)),
                            torch.max(high_ts, tsf.shift(close_ts, 1)))
        cond2 = tsf.diff(close_ts, 1) == 0
        inner = torch.where(cond2, zeros, inner)
        output_tensor = tsf.rolling_sum_(inner, 20)
        return output_tensor


class Alpha060(Functor):

    def forward(self, high_ts, low_ts, close_ts, volume_ts):
        output_tensor = tsf.rolling_sum_(((close_ts - low_ts) - (high_ts - close_ts)) / (high_ts - low_ts) * volume_ts,
                                         20)
        return output_tensor


class Alpha062(Functor):

    def forward(self, high_ts, volume_ts):
        output_tensor = (-1 * tsf.rolling_corr(high_ts, tsf.rank(volume_ts), 5))
        return output_tensor


class Alpha065(Functor):

    def forward(self, close_ts):
        output_tensor = tsf.rolling_mean_(close_ts, 6) / close_ts
        return output_tensor


class Alpha066(Functor):

    def forward(self, close_ts):
        output_tensor = (close_ts - tsf.rolling_mean_(close_ts, 6)) / tsf.rolling_mean_(close_ts, 6) * 100
        return output_tensor


class Alpha070(Functor):

    def forward(self, amount_ts):
        output_tensor = tsf.rolling_std(amount_ts, 6)
        return output_tensor


class Alpha071(Functor):

    def forward(self, close_ts):
        output_tensor = (close_ts - tsf.rolling_mean_(close_ts, 24)) / tsf.rolling_mean_(close_ts, 24) * 100
        return output_tensor


class Alpha074(Functor):

    def forward(self, vwap_ts, volume_ts, low_ts):
        output_tensor = (tsf.rank(tsf.rolling_corr(tsf.rolling_sum_(((low_ts * 0.35) + (vwap_ts * 0.65)), 20),
                                                   tsf.rolling_sum_(tsf.rolling_mean_(volume_ts, 40), 20),
                                                   7)) + tsf.rank(
            tsf.rolling_corr(tsf.rank(vwap_ts), tsf.rank(volume_ts), 6)))
        return output_tensor


class Alpha076(Functor):

    def forward(self, close_ts, volume_ts):
        output_tensor = tsf.rolling_std(torch.abs((close_ts / tsf.shift(close_ts, 1) - 1)) / volume_ts,
                                        20) / tsf.rolling_mean_(
            torch.abs((close_ts / tsf.shift(close_ts, 1) - 1)) / volume_ts, 20)
        return output_tensor


# class Alpha077(Functor): #decaylinear
#
#     def forward(self, high_ts, low_ts):
#         output_tensor = tsf.rolling_min(tsf.rank(DECAYLINEAR(((((HIGH	+	LOW)	/	2)	+	HIGH)	-	(VWAP	+	HIGH)),	20)),RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 3), 6)))
#         return output_tensor


class Alpha078(Functor):

    def forward(self, high_ts, low_ts, close_ts):
        output_tensor = ((high_ts + low_ts + close_ts) / 3 - tsf.rolling_mean_((high_ts + low_ts + close_ts) / 3,
                                                                               12)) / (0.015 * tsf.rolling_mean_(
            torch.abs(close_ts - tsf.rolling_mean_((high_ts + low_ts + close_ts) / 3, 12)), 12))
        return output_tensor


class Alpha083(Functor):

    def forward(self, high_ts, volume_ts):
        output_tensor = (-1 * tsf.rank(tsf.rolling_cov(tsf.rank(high_ts), tsf.rank(volume_ts), 5)))
        return output_tensor


class Alpha084(Functor):

    def forward(self, close_ts, volume_ts):
        sign_volume = torch.sign(tsf.diff(close_ts, 1)) * volume_ts
        output_tensor = tsf.rolling_sum_(sign_volume, 20)
        return output_tensor


class Alpha085(Functor):

    def forward(self, volume_ts, close_ts):
        output_tensor = (tsf.ts_rank((volume_ts / tsf.rolling_mean_(volume_ts, 20)), 20) * tsf.ts_rank(
            (-1 * tsf.diff(close_ts, 7)), 8))
        return output_tensor


class Alpha088(Functor):

    def forward(self, close_ts):
        output_tensor = (close_ts - tsf.shift(close_ts, 20)) / tsf.shift(close_ts, 20) * 100
        return output_tensor


class Alpha090(Functor):

    def forward(self, vwap_ts, volume_ts):
        output_tensor = (tsf.rank(tsf.rolling_corr(tsf.rank(vwap_ts), tsf.rank(volume_ts), 5)) * -1)
        return output_tensor


class Alpha091(Functor):

    def forward(self, close_ts, low_ts, volume_ts):
        output_tensor = ((tsf.rank((close_ts - tsf.rolling_max(close_ts, 5))) * tsf.rank(
            tsf.rolling_corr((tsf.rolling_mean_(volume_ts, 40)), low_ts, 5))) * -1)
        return output_tensor


class Alpha093(Functor):

    def forward(self, open_ts, low_ts):
        cond = open_ts >= tsf.shift(open_ts, 1)
        zeros = torch.zeros_like(open_ts)
        inner = torch.where(cond, zeros, torch.max((open_ts - low_ts), (open_ts - tsf.shift(open_ts, 1))))
        output_tensor = tsf.rolling_sum_(inner, 20)
        return output_tensor


class Alpha094(Functor):

    def forward(self, close_ts, volume_ts):
        sign_volume = torch.sign(tsf.diff(close_ts, 1)) * volume_ts
        output_tensor = tsf.rolling_sum_(sign_volume, 30)
        return output_tensor


class Alpha095(Functor):

    def forward(self, amount_ts):
        output_tensor = tsf.rolling_std(amount_ts, 20)
        return output_tensor


class Alpha097(Functor):

    def forward(self, volume_ts):
        output_tensor = tsf.rolling_std(volume_ts, 10)
        return output_tensor


class Alpha098(Functor):

    def forward(self, close_ts):
        cond = ((tsf.diff((tsf.rolling_sum_(close_ts, 100) / 100), 100) / tsf.shift(close_ts, 100)) < 0.05) | (
                (tsf.diff((tsf.rolling_sum_(close_ts, 100) / 100), 100) / tsf.shift(close_ts, 100)) == 0.05)
        consequence1 = (-1 * (close_ts - tsf.rolling_min(close_ts, 100)))
        consequence2 = (-1 * tsf.diff(close_ts, 3))
        output_tensor = torch.where(cond, consequence1, consequence2)
        return output_tensor


class Alpha100(Functor):

    def forward(self, volume_ts):
        output_tensor = tsf.rolling_std(volume_ts, 20)
        return output_tensor


# class Alpha101(Functor):
#
#     def forward(self, close_ts, high_ts, vwap_ts, volume_ts):
#         output_tensor = ((tsf.rank(
#             tsf.rolling_corr(close_ts, tsf.rolling_sum_(tsf.rolling_mean_(volume_ts, 30), 37), 15)) < tsf.rank(
#             tsf.rolling_corr(tsf.rank(((high_ts * 0.1) + (vwap_ts * 0.9))), tsf.rank(volume_ts), 11))) * -1)
#         return output_tensor


class Alpha103(Functor):

    def forward(self, low_ts):
        output_tensor = ((tsf.ts_argmin(low_ts, 20)+1) / 20) * 100
        return output_tensor


class Alpha104(Functor):

    def forward(self, high_ts, volume_ts, close_ts):
        output_tensor = (-1 * (
                tsf.diff(tsf.rolling_corr(high_ts, volume_ts, 5), 5) * tsf.rank(tsf.rolling_std(close_ts, 20))))
        return output_tensor


class Alpha105(Functor):

    def forward(self, open_ts, volume_ts):
        output_tensor = (-1 * tsf.rolling_corr(tsf.rank(open_ts), tsf.rank(volume_ts), 10))
        return output_tensor


class Alpha106(Functor):

    def forward(self, close_ts):
        output_tensor = tsf.diff(close_ts, 20)
        return output_tensor


class Alpha107(Functor):

    def forward(self, high_ts, low_ts, open_ts, close_ts):
        output_tensor = (((-1 * tsf.rank((open_ts - tsf.shift(high_ts, 1)))) * tsf.rank(
            (open_ts - tsf.shift(close_ts, 1)))) * tsf.rank((open_ts - tsf.shift(low_ts, 1))))
        return output_tensor


# class Alpha108(Functor):
#
#     def forward(self, high_ts, vwap_ts, volume_ts):
#         output_tensor = ((tsf.rank((high_ts - tsf.rolling_min(high_ts, 2))) ** tsf.rank(
#             tsf.rolling_corr((vwap_ts), (tsf.rolling_mean_(volume_ts, 120)), 6))) * -1)
#         return output_tensor


class Alpha110(Functor):

    def forward(self, high_ts, low_ts, close_ts):
        zeros = torch.zeros(high_ts.size())
        output_tensor = tsf.rolling_sum_(torch.max(zeros, high_ts - tsf.shift(close_ts, 1)), 20) / tsf.rolling_sum_(
            torch.max(zeros, tsf.shift(close_ts, 1) - low_ts), 20) * 100
        return output_tensor


class Alpha112(Functor):

    def forward(self, close_ts):
        inner1 = tsf.rolling_sum_(tsf.diff(close_ts, 12), 12)
        inner2 = tsf.rolling_sum_(torch.abs(tsf.diff(close_ts, 12)), 12)
        output_tensor = inner1 / inner2 * 100
        return output_tensor


class Alpha113(Functor):

    def forward(self, close_ts, volume_ts):
        output_tensor = (-1 * ((tsf.rank((tsf.rolling_sum_(tsf.shift(close_ts, 5), 20) / 20)) * tsf.rolling_corr(
            close_ts, volume_ts, 2)) * tsf.rank(
            tsf.rolling_corr(tsf.rolling_sum_(close_ts, 5), tsf.rolling_sum_(close_ts, 20), 2))))
        return output_tensor


class Alpha114(Functor):

    def forward(self, high_ts, low_ts, volume_ts, vwap_ts, close_ts):
        output_tensor = ((tsf.rank(tsf.shift(((high_ts - low_ts) / (tsf.rolling_sum_(close_ts, 5) / 5)), 2)) * tsf.rank(
            tsf.rank(volume_ts))) / (((high_ts - low_ts) / (tsf.rolling_sum_(close_ts, 5) / 5)) / (vwap_ts - close_ts)))
        return output_tensor


class Alpha115(Functor):

    def forward(self, high_ts, low_ts, close_ts, volume_ts):
        output_tensor = (tsf.rank(
            tsf.rolling_corr(((high_ts * 0.9) + (close_ts * 0.1)), tsf.rolling_mean_(volume_ts, 30), 10)) ** tsf.rank(
            tsf.rolling_corr(tsf.ts_rank(((high_ts + low_ts) / 2), 4), tsf.ts_rank(volume_ts, 10), 7)))
        return output_tensor


class Alpha117(Functor):

    def forward(self, high_ts, low_ts, volume_ts, close_ts, return_ts):
        output_tensor = ((tsf.ts_rank(volume_ts, 32) * (1 - tsf.ts_rank(((close_ts + high_ts) - low_ts), 16))) * (
                1 - tsf.ts_rank(return_ts, 32)))
        return output_tensor


class Alpha118(Functor):

    def forward(self, high_ts, low_ts, open_ts):
        output_tensor = tsf.rolling_sum_(high_ts - open_ts, 20) / tsf.rolling_sum_(open_ts - low_ts, 20) * 100
        return output_tensor


class Alpha120(Functor):

    def forward(self, vwap_ts, close_ts):
        output_tensor = (tsf.rank((vwap_ts - close_ts)) / tsf.rank((vwap_ts + close_ts)))
        return output_tensor


class Alpha121(Functor):

    def forward(self, volume_ts, vwap_ts):
        output_tensor = ((tsf.rank((vwap_ts - tsf.rolling_min(vwap_ts, 12))) ** tsf.ts_rank(
            tsf.rolling_corr(tsf.ts_rank(vwap_ts, 20), tsf.ts_rank(tsf.rolling_mean_(volume_ts, 60), 2), 18), 3)) * -1)
        return output_tensor


class Alpha126(Functor):

    def forward(self, high_ts, low_ts, close_ts):
        output_tensor = (close_ts + high_ts + low_ts) / 3
        return output_tensor


class Alpha127(Functor):

    def forward(self, close_ts):
        output_tensor = (tsf.rolling_mean_(
            (100 * (close_ts - tsf.rolling_max(close_ts, 12)) / (tsf.rolling_max(close_ts, 12))) ** 2)) ** (1 / 2)
        return output_tensor


# class Alpha128(Functor):
#
#     def forward(self, high_ts, low_ts, close_ts, volume_ts):
#         zeros = torch.zeros(high_ts.size())
#         cond1 = (high_ts + low_ts + close_ts) / 3 > tsf.shift((high_ts + low_ts + close_ts) / 3, 1)
#         inner1 = torch.where(cond1, (high_ts + low_ts + close_ts) / 3 * volume_ts, zeros)
#         inner1 = 1 + tsf.rolling_sum_(inner1, 14)
#         cond2 = (high_ts + low_ts + close_ts) / 3 < tsf.shift((high_ts + low_ts + close_ts) / 3, 1)
#         inner2 = torch.where(cond2, (high_ts + low_ts + close_ts) / 3 * volume_ts, zeros)
#         inner2 = tsf.rolling_sum_(inner2, 12)
#         output_tensor = 100 - (100 / inner1 / inner2)
#         return output_tensor


class Alpha129(Functor):

    def forward(self, high_ts, close_ts):
        zeros = torch.zeros_like(high_ts)
        cond = tsf.diff(close_ts, 1) < 0
        output_tensor = tsf.rolling_sum_(torch.where(cond, torch.abs(tsf.diff(close_ts, 1)), zeros), 12)
        return output_tensor


# class Alpha131(Functor):
#
#     def forward(self, vwap_ts, volume_ts, close_ts):
#         output_tensor = (tsf.rank(tsf.shift(vwap_ts, 1)) ** tsf.ts_rank(
#             tsf.rolling_corr(close_ts, tsf.rolling_mean_(volume_ts, 50), 18), 18))
#         return output_tensor


class Alpha132(Functor):

    def forward(self, amount_ts):
        output_tensor = tsf.rolling_mean_(amount_ts, 20)
        return output_tensor


class Alpha133(Functor):

    def forward(self, high_ts, low_ts):
        output_tensor = ((tsf.ts_argmax(high_ts)+1) / 20) * 100 - (((tsf.ts_argmin(low_ts))+1 )/ 20) * 100
        return output_tensor


class Alpha134(Functor):

    def forward(self, volume_ts, close_ts):
        output_tensor = (close_ts - tsf.shift(close_ts, 12)) / tsf.shift(close_ts, 12) * volume_ts
        return output_tensor


class Alpha136(Functor):

    def forward(self, volume_ts, open_ts, return_ts):
        output_tensor = ((-1 * tsf.rank(tsf.diff(return_ts, 3))) * tsf.rolling_corr(open_ts, volume_ts, 10))
        return output_tensor


class Alpha139(Functor):

    def forward(self, open_ts, volume_ts):
        output_tensor = (-1 * tsf.rolling_corr(open_ts, volume_ts, 10))
        return output_tensor


class Alpha141(Functor):

    def forward(self, high_ts, volume_ts):
        output_tensor = (
                tsf.rank(tsf.rolling_corr(tsf.rank(high_ts), tsf.rank(tsf.rolling_mean_(volume_ts, 15)), 9)) * -1)
        return output_tensor


class Alpha142(Functor):

    def forward(self, volume_ts, close_ts):
        output_tensor = (((-1 * tsf.rank(tsf.ts_rank(close_ts, 10))) * tsf.rank(
            tsf.diff(tsf.diff(close_ts, 1), 1))) * tsf.rank(
            tsf.ts_rank((volume_ts / tsf.rolling_mean_(volume_ts, 20)), 5)))
        return output_tensor


class Alpha145(Functor):

    def forward(self, high_ts, low_ts, close_ts):
        zeros = torch.zeros(high_ts.size())
        output_tensor = tsf.rolling_sum_(torch.max(zeros, high_ts - tsf.shift(close_ts, 1)), 20) / tsf.rolling_sum_(
            torch.max(zeros, tsf.shift(close_ts, 1) - low_ts), 20) * 100
        return output_tensor


class Alpha150(Functor):

    def forward(self, high_ts, low_ts, close_ts, volume_ts):
        output_tensor = (high_ts + low_ts + close_ts) / 3 * volume_ts
        return output_tensor


class Alpha159(Functor):

    def forward(self, high_ts, close_ts, low_ts):
        output_tensor = (close_ts - tsf.rolling_sum_(torch.min(low_ts, tsf.shift(close_ts, 1)), 6)) / tsf.rolling_sum_(
            torch.max(high_ts, tsf.shift(close_ts, 1)) - torch.min(low_ts, tsf.shift(close_ts, 1)),
            6) * 12 * 24 + (
                                close_ts - tsf.rolling_sum_(torch.min(low_ts, tsf.shift(close_ts, 1)),
                                                            12)) / tsf.rolling_sum_(
            torch.max(high_ts, tsf.shift(close_ts, 1)) - torch.min(low_ts, tsf.shift(close_ts, 1)), 12) * 6 * 24 + (
                                close_ts - tsf.rolling_sum_(torch.min(low_ts, tsf.shift(close_ts, 1)),
                                                            24)) / tsf.rolling_sum_(
            torch.max(high_ts, tsf.shift(close_ts, 1)) - torch.min(low_ts, tsf.shift(close_ts, 1)),
            24) * 6 * 24 * 100 / (6 * 12 + 6 * 24 + 12 * 24)
        return output_tensor


class Alpha161(Functor):

    def forward(self, high_ts, low_ts, close_ts):
        output_tensor = tsf.rolling_mean_(
            torch.max(torch.max((high_ts - low_ts), torch.abs(tsf.shift(close_ts, 1) - high_ts)),
                      torch.abs(tsf.shift(close_ts, 1) - low_ts)), 12)
        return output_tensor


class Alpha163(Functor):

    def forward(self, high_ts, return_ts, close_ts, vwap_ts, volume_ts):
        output_tensor = tsf.rank(
            ((((-1 * return_ts) * tsf.rolling_mean_(volume_ts, 20)) * vwap_ts) * (high_ts - close_ts)))
        return output_tensor


class Alpha167(Functor):

    def forward(self, close_ts):
        cond = tsf.diff(close_ts, 1) > 0
        zeros = torch.zeros_like(close_ts)
        output_tensor = tsf.rolling_sum_(torch.where(cond, close_ts - tsf.shift(close_ts, 1), zeros), 12)
        return output_tensor


class Alpha168(Functor):

    def forward(self, volume_ts):
        output_tensor = (-1 * volume_ts / tsf.rolling_mean_(volume_ts, 20))
        return output_tensor


class Alpha170(Functor):

    def forward(self, high_ts, close_ts, volume_ts, vwap_ts):
        output_tensor = ((((tsf.rank((1 / close_ts)) * volume_ts) / tsf.rolling_mean_(volume_ts, 20)) * (
                (high_ts * tsf.rank((high_ts - close_ts))) / (tsf.rolling_sum_(high_ts, 5) / 5))) - tsf.rank(
            (vwap_ts - tsf.shift(vwap_ts, 5))))
        return output_tensor


class Alpha171(Functor):

    def forward(self, high_ts, low_ts, close_ts, open_ts):
        output_tensor = ((-1 * ((low_ts - close_ts) * (open_ts ** 5))) / ((close_ts - high_ts) * (close_ts ** 5)))
        return output_tensor


class Alpha172(Functor):

    def forward(self, high_ts, low_ts, close_ts):
        zeros = torch.zeros_like(high_ts)
        LD = tsf.shift(low_ts, 1) - low_ts
        HD = high_ts - tsf.shift(high_ts, 1)
        TR = torch.max(torch.max(high_ts - low_ts, torch.abs(high_ts - tsf.shift(close_ts, 1))),
                       torch.abs(low_ts - tsf.shift(close_ts, 1)))
        cond1 = (LD > 0) & (LD > HD)
        inner1 = torch.where(cond1, LD, zeros)
        cond2 = (HD > 0) & (HD > LD)
        inner2 = torch.where(cond2, HD, zeros)
        cond3 = (LD > 0) & (LD > HD)
        inner3 = torch.where(cond3, LD, zeros)
        cond4 = (HD > 0) & (HD > LD)
        inner4 = torch.where(cond4, HD, zeros)
        output_tensor = tsf.rolling_mean_(
            tsf.rolling_sum_(inner1, 14) * 100 / tsf.rolling_sum_(TR, 14) - tsf.rolling_sum_(inner2,
                                                                                             14) * 100 / tsf.rolling_sum_(
                TR, 14) / (
                    tsf.rolling_sum_(inner3, 14) * 100 / tsf.rolling_sum_(TR, 14) + tsf.rolling_sum_(inner4,
                                                                                                     14) * 100 / tsf.rolling_sum_(
                TR, 14)) * 100, 6)
        return output_tensor


class Alpha175(Functor):

    def forward(self, high_ts, low_ts, close_ts):
        output_tensor = tsf.rolling_mean_(
            torch.max(torch.max((high_ts - low_ts), torch.abs(tsf.shift(close_ts, 1) - high_ts)),
                      torch.abs(tsf.shift(close_ts, 1) - low_ts)), 6)
        return output_tensor


class Alpha176(Functor):

    def forward(self, high_ts, low_ts, close_ts, volume_ts):
        output_tensor = tsf.rolling_corr(tsf.rank(
            ((close_ts - tsf.rolling_min(low_ts, 12)) / (tsf.rolling_max(high_ts, 12) - tsf.rolling_min(low_ts, 12)))),
            tsf.rank(volume_ts), 6)
        return output_tensor


class Alpha177(Functor):

    def forward(self, high_ts):
        output_tensor = ((tsf.ts_argmax(high_ts, 20)+1) / 20) * 100
        return output_tensor


class Alpha178(Functor):

    def forward(self, volume_ts, close_ts):
        output_tensor = (close_ts - tsf.shift(close_ts, 1)) / tsf.shift(close_ts, 1) * volume_ts
        return output_tensor


class Alpha179(Functor):

    def forward(self, vwap_ts, low_ts, volume_ts):
        output_tensor = (tsf.rank(tsf.rolling_corr(vwap_ts, volume_ts, 4)) * tsf.rank(
            tsf.rolling_corr(tsf.rank(low_ts), tsf.rank(tsf.rolling_mean_(volume_ts, 50)), 12)))
        return output_tensor


# class Alpha180(Functor): #sequance1 failed
#
#     def forward(self, volume_ts, close_ts):
#         cond = (tsf.rolling_mean_(volume_ts, 20) < volume_ts)
#         output_tensor = torch.where(cond, ((-1 * tsf.ts_rank(torch.abs(tsf.diff(close_ts, 7)), 60)) * torch.sign(
#             tsf.diff(close_ts, 7))).float(),
#                                     -1 * volume_ts)
#         return output_tensor


class Alpha185(Functor):

    def forward(self, high_ts, open_ts, close_ts):
        output_tensor = (tsf.rank(tsf.rolling_corr(tsf.shift((open_ts - close_ts), 1), close_ts, 200)) + tsf.rank(
            (open_ts - close_ts)))
        return output_tensor


class Alpha186(Functor):

    def forward(self, open_ts, close_ts):
        output_tensor = tsf.rank((-1 * ((1 - (open_ts / close_ts)) ** 2)))
        return output_tensor


class Alpha187(Functor):

    def forward(self, close_ts, low_ts, high_ts):
        zeros = torch.zeros_like(low_ts)
        LD = tsf.shift(low_ts, 1) - low_ts
        HD = high_ts - tsf.shift(high_ts, 1)
        TR = torch.max(torch.max(high_ts - low_ts, torch.abs(high_ts - tsf.shift(close_ts, 1))),
                       torch.abs(low_ts - tsf.shift(close_ts, 1)))
        cond1 = (LD > 0) & (LD > HD)
        inner1 = torch.where(cond1, LD, zeros)
        cond2 = (HD > 0) & (HD > LD)
        inner2 = torch.where(cond2, HD, zeros)
        cond3 = (LD > 0) & (LD > HD)
        inner3 = torch.where(cond3, LD, zeros)
        cond4 = (HD > 0) & (HD > LD)
        inner4 = torch.where(cond4, HD, zeros)
        output_tensor = (tsf.rolling_mean_(torch.abs(tsf.rolling_sum_(inner1, 14) * 100 / tsf.rolling_sum_(TR, 14) - tsf.rolling_sum_(inner2, 14) * 100 / tsf.rolling_sum_(TR,14)) /tsf.rolling_sum_(inner3, 14) * 100 / tsf.rolling_sum_(TR, 14) + tsf.rolling_sum_(inner4,14) * 100 / tsf.rolling_sum_(TR,14) * 100,6) + tsf.shift(
            tsf.rolling_mean_(torch.abs(
                tsf.rolling_sum_(inner1, 14) * 100 / tsf.rolling_sum_(TR, 14) - tsf.rolling_sum_(inner2,
                                                                                                 14) * 100 / tsf.rolling_sum_(
                    TR, 14)) / tsf.rolling_sum_(inner3, 14) * 100 / tsf.rolling_sum_(TR, 14) + tsf.rolling_sum_(
                inner4, 14) * 100 / tsf.rolling_sum_(TR, 14) * 100, 6), 6)) / 2
        return output_tensor


class Alpha188(Functor):

    def forward(self, open_ts, high_ts):
        zeros = torch.zeros_like(open_ts)
        cond = open_ts <= tsf.shift(open_ts, 1)
        inner = torch.where(cond, zeros, torch.max((high_ts - open_ts), (open_ts - tsf.shift(open_ts, 1))))
        output_tensor = tsf.rolling_sum_(inner, 20)
        return output_tensor


class Alpha190(Functor):

    def forward(self, close_ts):
        output_tensor = tsf.rolling_mean_(torch.abs(close_ts - tsf.rolling_mean_(close_ts, 6)), 6)
        return output_tensor


class Alpha192(Functor):

    def forward(self, close_ts, volume_ts, low_ts, high_ts):
        output_tensor = ((tsf.rolling_corr(tsf.rolling_mean_(volume_ts, 20), low_ts, 5) + (
                (high_ts + low_ts) / 2)) - close_ts)
        return output_tensor

