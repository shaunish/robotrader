import datetime as dt
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import pandas as pd

def vol(cp, window=15):
    cp['vol'] = cp['Close'].rolling(window=window).apply(np.std)
    return cp

def momentum(cp, window=5):
    cp['shifted'] = cp['Close'].copy().shift(periods=window)
    cp['Momentum'] = (cp['Close'] - cp['shifted']) / cp['shifted']
    cp.drop(columns=['shifted'], inplace=True)
    return cp

def percentage_price(cp):
    alpha26 = 2 / 27
    alpha9 = .2
    def ema(cp, alpha):
        df = cp['Close'].copy()
        ema = [df.iloc[0]]
        for row in df:
            val = row * alpha + (1 - alpha)*ema[-1]
            ema.append(val)
        return np.array(ema[1:])
    cp['ema26'] = ema(cp, alpha26)
    cp['ema9'] = ema(cp, alpha9)
    cp['ppi'] = (cp['ema9'] - cp['ema26']) / cp['ema26']
    cp.drop(columns=['ema26', 'ema9'], inplace=True)
    return cp

def aroon(cp, window=30):
    max_pos = cp['Close'].rolling(window=window).apply(np.argmax, raw=True)
    cp['aroon_up'] = (window - max_pos - 1) / window
    min_pos = cp['Close'].rolling(window=window).apply(np.argmin, raw=True)
    cp['aroon_down'] = (window - min_pos - 1) / window
    cp['aroon'] = cp['aroon_up'] - cp['aroon_down']
    return cp.drop(columns=['aroon_up', 'aroon_down'])

def mfi(cp, window=30):
    moneyflow = cp['Close'] * cp['Volume']
    # mask = cp['Close'] - cp['Close'].shift(1) > 0
    #print(moneyflow[mask].rolling(window=window).sum())
    pmf = moneyflow[cp['Close'] - cp['Close'].shift(1) > 0]
    pmf.name = 'pmf'
    nmf = moneyflow[cp['Close'] - cp['Close'].shift(1) < 0]
    nmf.name = 'nmf'
    moneyflow = pd.DataFrame(moneyflow)
    moneyflow = moneyflow.merge(pmf, how='left', left_index=True, right_index=True)
    moneyflow = moneyflow.merge(nmf, how='left', left_index=True, right_index=True)
    moneyflow = moneyflow.fillna(0)
    moneyflow['tpmf'] = moneyflow['pmf'].rolling(window=window).sum()
    moneyflow['tnmf'] = moneyflow['nmf'].rolling(window=window).sum()
    moneyflow['mfi'] = moneyflow['tpmf'] / (moneyflow['tpmf']  + moneyflow['tnmf'])
    # moneyflow.drop(columns=['pmf', 'nmf', 'tpmf','tnmf'], inplace=True)
    cp['mfi'] = moneyflow['mfi']
    return cp


