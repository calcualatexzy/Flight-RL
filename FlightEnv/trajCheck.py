import numpy as np

class TrajCheck:
    def __init__(self, trajData):
        self.dimension = 15
        self.checkData = trajData
        self.eigenValues = []
        self.eigenVectors = []
        self.meanVector = np.zeros(self.dimension)
        self.covMatrix = np.zeros((self.dimension, self.dimension))

    def generateCovMatrix(self):
        self.covMatrix = np.cov(self.checkData, rowvar=False) 
    
    def updateMeanVector(self):
        self.meanVector = np.mean(self.checkData, axis=0)
    
    def updateMeanVectorAndCovMatrix(self, newVector):   #新加入一个向量，用递推形式更新协方差矩阵
        #newVector = np.array(newVector)
        oldMeanVector = self.meanVector.copy()
        #self.checkData = np.vstack([self.checkData, newVector])
        self.checkData.append(newVector)
        self.meanVector = np.mean(self.checkData, axis=0)
        self.covMatrix = self.covMatrix * (len(self.checkData) - 2) / (len(self.checkData) - 1) + np.outer(newVector - oldMeanVector, newVector - self.meanVector) / (len(self.checkData) - 1)
    
    def computeEigenvalues(self):
        if self.covMatrix is not None:
            self.eigenValues, _ = np.linalg.eigh(self.covMatrix)
            self.eigenValues = sorted(self.eigenValues, reverse=True)
        
    def computeContributions(self):
        total = sum(self.eigenValues)
        contributions = [eigenValue / total for eigenValue in self.eigenValues]
        return contributions
    
    def jugdeDivergence(self, num):    #检查特征值贡献是否平均
        list = self.computeContributions()
        if(list[0] >= 0.3):
            return False
        cnt = 0
        for p in list:
            if p > 0.3 / self.dimension:
                cnt += 1
        if cnt >= num:
            return True
        else:
            return False


if __name__ == '__main__':
    trajCheck = TrajCheck()
    trajCheck.updateMeanVector()
    trajCheck.generateCovMatrix()
    print(trajCheck.meanVector)  
    print(trajCheck.covMatrix)    
    trajCheck.computeEigenvalues()
    print(trajCheck.eigenValues) 
    print(trajCheck.computeContributions() )