from DLNest.Common.DatasetBase import DatasetBase
from DLNest.Common.ModelBase import ModelBase
from DLNest.Common.LifeCycleBase import LifeCycleBase


class LifeCycle(LifeCycleBase):
    def BAll(self):
        self.best_FPD = 10000
    
    def BTrain(self):
        print("Start training")
    
    def getSaveDict(self):
        return {
            "best_FPD" : self.best_FPD
        }
    
    def loadSaveDict(self, saveDict):
        self.best_FPD = saveDict["best_FPD"]

    def needVisualize(self, epoch : int, iter : int, logdict : dict, args : dict):
        return True

    def needValidation(self, epoch : int, logdict : dict, args : dict):
        return args["model_config"]["need_FPD"]

    def commandLineOutput(self,epoch : int, logdict : dict, args : dict):
        try:
            print("Epoch #" + str(epoch) + " finished!")
            print("Last :",logdict["FPD"][-1],"Best :",self.best_FPD)
        except Exception:
            pass

    def needSaveModel(self, epoch : int, logdict : dict, args : dict):
        return True

    def holdThisCheckpoint(self, epoch : int, logdict : dict, args : dict):
        if args["model_config"]["need_FPD"]:
            if logdict["FPD"][-1] < self.best_FPD:
                self.best_FPD = logdict["FPD"][-1]
                return True
        else:
            return False

    def needContinueTrain(self, epoch : int, logdict : dict, args : dict):
        if epoch < args["epochs"]:
            return True
        return False
