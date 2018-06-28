from collections import OrderedDict
import numpy as np
import math

from matplotlib import pyplot as plt

import OpenCOR as oc
times = np.array([0, 240, 480, 960, 1920, 3840])
pFC = np.array([0.0, 0.0189, 0.0208, 0.0646, 0.0495, 0.0645])
pSyk = np.array([0.0, 0.0143, 0.0255, 0.0303, 0.0242, 0.0202])
#f=open("kf.txt",'w')  
class Simulation(object):
     def __init__(self):
         self.simulation = oc.simulation()
         self.simulation.data().setStartingPoint(0)
         self.simulation.data().setEndingPoint(3840)
         self.simulation.data().setPointInterval(1)
         self.constants = self.simulation.data().constants()
         for k,v in dict.items():
              self.constants[k]=v
         self.model_constants = OrderedDict({k: self.constants[k]
                                            for k in self.constants.keys()})            
         
     def run_once(self, c, v):
         print (c)
         print (v)
         self.simulation.resetParameters()
         self.constants[c] = v
         self.simulation.run()
         return (self.simulation.results().points().values(),
                 self.simulation.results().states()['FCepsilonRI/pFC'].values())
     
     def run(self, c, scale=2.0):
         self.simulation.clearResults()
         v = self.model_constants[c]
         base = self.run_once(c, v)[1][times]
         divergence = 0.0
         error = 0.0
         #print(base)
         #print (len(base))
         #plt.plot(times, base, '-')
         #for s in [1.0/scale, scale]:
         #    trial = self.run_once(c, s*v)[1][times]
         #    divergence += math.sqrt(np.sum((base - trial)**2))
         divergence +=  math.sqrt(np.sum((base - pFC)**2))
         error += np.sum((base - pFC)**2)
         error=round(error, 6)
         print(error)
         #print ('1',error)
         #print ('2',self.constants['FCepsilonRI/k_f1'])
         #f.write(" %s  %s \n" %(self.constants['FCepsilonRI/k_f5'],error))
         
         #Plot graphs            
         plt.plot(self.constants['FCepsilonRI/k_f5'], error, '*')
         plt.hold()      # toggle hold
         plt.hold(True)
         return divergence

     def test_run(self):
         variation = OrderedDict()
         for c in self.model_constants.keys():
             if c=='FCepsilonRI/k_f5':
                variation[c] = self.run(c)
         return variation
     
     def test(self, c, v):
         trial = self.run_once(c, s*v)[1][times]
         return math.sqrt(np.sum((pFC - trial)**2))
                              
dict = {'FCepsilonRI/k_f1': 1, 'FCepsilonRI/k_r1': 0, 'FCepsilonRI/k_f2': 1, 'FCepsilonRI/k_f3': 1,
        'FCepsilonRI/k_r3': 0, 'FCepsilonRI/k_f4': 1, 'FCepsilonRI/k_f5': 1, 'FCepsilonRI/k_r5': 0}

for i in np.arange (0.0009,101,1):
      dict['FCepsilonRI/k_f5'] = i
      
      s = Simulation()
      
      v = s.test_run()
        
plt.show()
#print({ k:d  for k, d in v.items() if d > 0.001 })
#f.close()  


    
    
