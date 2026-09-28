import numpy as np
import sys
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from log_file import setup_logging
logger = setup_logging("main")
from variable_transformation_data import varaible_transform
from column_selection import cloumns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from all_models import aoc_roc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
import pickle
class Main:
    def __init__(self,d):
        try:
            df = pd.read_csv(d)
            # df = df.drop_duplicates()          #removing all the duplicate values
            # df.reset_index(drop=True, inplace=True)     #resetting the index
            X=df.iloc[:,:-1]
            y=df.iloc[:,-1]
            self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(X,y,test_size=0.2,random_state=42)
        except:
            er_msg,er_type,er_line=sys.exc_info()
            logger.info(f"error type {er_type}error in line {er_line.tb_lineno} due to {er_msg}")


    #   Applying variable transformation and outliers technique for removine the outliers so that all data will be safe side


    def variable_transformation(self):
        try:
            logger.info(f"before performing variable transformation and  outlier:\n \n{self.X_train}")
            logger.info(f"before performing variable transformation and  outlier:\n \n{self.X_test}")

            # Calling the function where variable transformation and outliers are performed

            self.X_train,self.X_test=varaible_transform(self.X_train, self.X_test)
            logger.info(f"after performing variable transformation and  outlier:\n \n{self.X_train}")
            logger.info(f"after performing variable transformation and  outlier:\n \n{self.X_test}")
        except:
            er_msg, er_type, er_line = sys.exc_info()
            logger.info(f"error type {er_type}error in line {er_line.tb_lineno} due to {er_msg}")

    #   Selecting the best columns using Feature selection methods

    def feature_selection(self):
        try:
            logger.info(f"before performing feature_selection:\n \n{self.X_train}")
            logger.info(f"before performing feature_selection:\n \n{self.X_test}")

            #calling the function where feature_selection happens

            self.X_train,self.X_test=cloumns(self.X_train,self.X_test,self.y_train,self.y_test)
            logger.info(f"after performing feature_selection:\n \n{self.X_train}")
            logger.info(f"after performing feature_selection:\n \n{self.X_test}")
        except:
            er_msg, er_type, er_line = sys.exc_info()
            logger.info(f"error type {er_type}error in line {er_line.tb_lineno} due to {er_msg}")


    #Balancing the data and scaling the data for balancing learning accuracy of the model


    def balencing_data(self):
        try:
            reg=SMOTE(random_state=42)
            self.X_train,self.y_train=reg.fit_resample(self.X_train,self.y_train)
            logger.info(f"after balancing:\n \n{self.X_train}")
            logger.info(f"after balancing:\n \n{sum(self.y_train==1)}")
            logger.info(f"after balancing:\n \n{sum(self.y_train==0)}")

             #now lets scale down values

            logger.info(f"before scaling down:\n \n{self.X_train}")
            logger.info(f"before scaling down:\n \n{self.X_test}")
            self.st_obj=StandardScaler()
            self.st_obj.fit(self.X_train)
            self.X_train_bal=self.st_obj.transform(self.X_train)
            self.X_test_bal=self.st_obj.transform(self.X_test)
            logger.info(f"after scaling down:\n \n{self.X_train}")
            logger.info(f"after scaling down:\n \n{self.X_test}")


        except:
            er_msg, er_type, er_line = sys.exc_info()
            logger.info(f"error type {er_type}error in line {er_line.tb_lineno} due to {er_msg}")

    def train_all_models(self):
        try:
            # aoc_roc(self.X_train ,self.y_train, self.X_test, self.y_test)
            self.nb_reg = GaussianNB()
            self.nb_reg.fit(self.X_train_bal, self.y_train)
            logger.info(f"Test Data Accuracy : {accuracy_score(self.y_test, self.nb_reg.predict(self.X_test_bal))}")
            logger.info(f"Test Data Confusion Matrix : {confusion_matrix(self.y_test, self.nb_reg.predict(self.X_test_bal))}")
            logger.info(f"Test Data Classification Report  : {classification_report(self.y_test, self.nb_reg.predict(self.X_test_bal))}")

        #     # after perfoforming i get to know that KNNbgive good performace for my model
        #
        #     knn_obj = KNeighborsClassifier(n_neighbors=5)
        #     knn_obj.fit(self.X_train, self.y_train)
        #     logger.info(f"Test Data Accuracy :\n \n {accuracy_score(self.y_test, knn_obj.predict(self.X_test))}")
        #     logger.info(f"Test Data Confusion Matrix :\n \n {confusion_matrix(self.y_test, knn_obj.predict(self.X_test))}")
        #     logger.info(f"Test Data Classification Report  :\n \n {classification_report(self.y_test, knn_obj.predict(self.X_test))}")
        #
           #performing hyper parameter tuning
        #
            parameters = {"var_smoothing" : np.logspace(-10,-1,10)}
            g = GridSearchCV(
                estimator=self.nb_reg,
                param_grid=parameters,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            g.fit(self.X_train_bal, self.y_train)

            logger.info(f"Best Parameters:{ g.best_params_}")
            logger.info(f"Best Score: {g.best_score_}")

             # after performing hyper_parameter tuning I got best parameters for my so that i can improve it accuracy
            nb_reg = GaussianNB(var_smoothing= np.float64(1e-10))
            nb_reg.fit(self.X_train_bal, self.y_train)
            logger.info(f"Test Data Accuracy : {accuracy_score(self.y_test, nb_reg.predict(self.X_test_bal))}")
            logger.info(f"Test Data Confusion Matrix : {confusion_matrix(self.y_test, nb_reg.predict(self.X_test_bal))}")
            logger.info(f"Test Data Classification Report  : {classification_report(self.y_test, nb_reg.predict(self.X_test_bal))}")

        #
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")
    def testing_new_data_and_saving_model(self):
        try:
            a=np.array([[60, 1,3,172,0.6 , 0 , 2]])
            self.st_obj.transform(a)
            logger.info(f"predicted value{self.nb_reg.predict(a)}")
            with open("model.pkl","wb") as p:
                pickle.dump(self.nb_reg,p)
            with open('scaled.pkl', 'wb') as t:
                pickle.dump(self.st_obj,t)

        except:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")
if __name__=="__main__":
    try:
        obj=Main("D:\ML_viharatech projects\mini_project-2\heart.csv")
        obj.variable_transformation()
        obj.feature_selection()
        obj.balencing_data()
        obj.train_all_models()
        obj.testing_new_data_and_saving_model()
    except:
        er_msg, er_type, er_line = sys.exc_info()
        logger.info(f"error type {er_type}error in line {er_line.tb_lineno} due to {er_msg}")
