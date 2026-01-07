import math

def learning_rate_scheduling(t,
                             a_max,
                             a_min,
                             T_w,
                             T_c):

    #  Ift<Tw,thenαt= t/Tw * αmax
    if t<T_w:
        a_t = t/T_w * a_max
        return a_t 
      
    #  If Tw≤t≤Tc,then αt=αmin +1/2(1+cos(t−Tw/Tc−Tw)π))(αmax−αmin).
    if T_w<=t and t<=T_c:
        theta = ((t-T_w)/(T_c-T_w))*math.pi
        a_t = a_min + (1/2)*(1+ math.cos(theta)) * (a_max-a_min)
        return a_t
      
    # If t>Tc,then αt=αmin.
    if t>T_c:
        a_t = a_min
        return a_t
      
