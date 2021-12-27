#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
import argparse
import datetime

import backtrader as bt


class St(bt.Strategy):
    params = dict(
        indicators=False,
        indperiod1=10,
        indperiod2=50,
        indicator=bt.ind.SMA,
        trade=False,
    )

    def __init__(self):
        self.dtinit = datetime.datetime.now()
        print('Strat Init Time:             {}'.format(self.dtinit))
        loaddata = (self.dtinit - self.env.dtcerebro).total_seconds()
        print('Time Loading Data Feeds:     {:.2f}'.format(loaddata))

        print('Number of data feeds:        {}'.format(len(self.datas)))
        if self.p.indicators:
            total_ind = self.p.indicators * 3 * len(self.datas)
            print('Total indicators:            {}'.format(total_ind))
            indname = self.p.indicator.__name__
            print('Moving Average to be used:   {}'.format(indname))
            print('Indicators period 1:         {}'.format(self.p.indperiod1))
            print('Indicators period 2:         {}'.format(self.p.indperiod2))

            self.macross = {}
            for d in self.datas:
                ma1 = self.p.indicator(d, period=self.p.indperiod1)
                ma2 = self.p.indicator(d, period=self.p.indperiod2)
                self.macross[d] = bt.ind.CrossOver(ma1, ma2)

    def start(self):
        self.dtstart = datetime.datetime.now()
        print('Strat Start Time:            {}'.format(self.dtstart))

    def prenext(self):
        if len(self.data0) == 1:  # only 1st time
            self.dtprenext = datetime.datetime.now()
            print('Pre-Next Start Time:         {}'.format(self.dtprenext))
            indcalc = (self.dtprenext - self.dtstart).total_seconds()
            print('Time Calculating Indicators: {:.2f}'.format(indcalc))

    def nextstart(self):
        if len(self.data0) == 1:  # there was no prenext
            self.dtprenext = datetime.datetime.now()
            print('Pre-Next Start Time:         {}'.format(self.dtprenext))
            indcalc = (self.dtprenext - self.dtstart).total_seconds()
            print('Time Calculating Indicators: {:.2f}'.format(indcalc))

        self.dtnextstart = datetime.datetime.now()
        print('Next Start Time:             {}'.format(self.dtnextstart))
        warmup = (self.dtnextstart - self.dtprenext).total_seconds()
        print('Strat warm-up period Time:   {:.2f}'.format(warmup))
        nextstart = (self.dtnextstart - self.env.dtcerebro).total_seconds()
        print('Time to Strat Next Logic:    {:.2f}'.format(nextstart))
        self.next()

    def next(self):
        if not self.p.trade:
            return

        for d, macross in self.macross.items():
            if macross > 0:
                self.order_target_size(data=d, target=1)
            elif macross < 0:
                self.order_target_size(data=d, target=-1)

    def stop(self):
        dtstop = datetime.datetime.now()
        print('End Time:                    {}'.format(dtstop))
        nexttime = (dtstop - self.dtnextstart).total_seconds()
        print('Time in Strategy Next Logic: {:.2f}'.format(nexttime))
        strattime = (dtstop - self.dtprenext).total_seconds()
        print('Total Time in Strategy:      {:.2f}'.format(strattime))
        totaltime = (dtstop - self.env.dtcerebro).total_seconds()
        print('Total Time:                  {:.2f}'.format(totaltime))
        print('Length of data feeds:        {}'.format(len(self.data)))


def run(args=None):
    args = parse_args(args)

    cerebro = bt.Cerebro()

    datakwargs = dict(timeframe=bt.TimeFrame.Minutes, compression=15)
    for i in range(args.numfiles):
        dataname = 'candles{:02d}.csv'.format(i)
        data = bt.feeds.GenericCSVData(dataname=dataname, **datakwargs)
        cerebro.adddata(data)

    cerebro.addstrategy(St, **eval('dict(' + args.strat + ')'))
    cerebro.dtcerebro = dt0 = datetime.datetime.now()
    print('Cerebro Start Time:          {}'.format(dt0))
    cerebro.run(**eval('dict(' + args.cerebro + ')'))


def parse_args(pargs=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            'Backtrader Basic Script'
        )
    )

    parser.add_argument('--numfiles', required=False, default=100, type=int,
                        help='Number of files to rea')

    parser.add_argument('--cerebro', required=False, default='',
                        metavar='kwargs', help='kwargs in key=value format')

    parser.add_argument('--strat', '--strategy', required=False, default='',
                        metavar='kwargs', help='kwargs in key=value format')


    return parser.parse_args(pargs)


if __name__ == '__main__':
    run()
