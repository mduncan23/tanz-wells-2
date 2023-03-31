import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score, classification_report, auc, \
plot_roc_curve, confusion_matrix, plot_confusion_matrix

from sklearn.model_selection import train_test_split, cross_validate,\
cross_val_score, RandomizedSearchCV, GridSearchCV


class ModelRuns():
    ''' Class to easily run models through and determine output.'''
    
    def __init__(self, model, model_name, X, y):
        self.model = model
        self.name = model_name
        self.X = X
        self.y = y
        
    def cross_val(self, X=None, y=None, cv=5):
        cv_X= X if X else self.X
        cv_y = y if y else self.y
        self.cv_results = cross_val_score(self.model, cv_X, cv_y, cv=cv)
        self.cv_mean = np.mean(self.cv_results)
        return print(f"Cross_val mean is: {round(self.cv_mean, 4)} for kfold: {cv}")
                
    def accuracy(self, X, y):
        acc = self.model.score(X, y)
        
    def performance(self, X, y):
        y_pred = self.model.predict(X)
        report = classification_report(y, y_pred)
        return report
    
    def class_matrix(self, X, y):
        mat = plot_confusion_matrix(self.model, X, y);
        if X is self.X_train_run:
            plt.title("Train")
        elif X is self.X_test_run:
            plt.title('Test')
        else:
            plt.title('Class Matrix')
        return mat
    
    def roc_auc_custom(self, X, y):
        y_score = self.model.predict_proba(X)
        roc_score = roc_auc_score(y, y_score[:,1])
        return roc_score
    
    def run_all(self, X_train_run=None, X_test_run=None, y_train_run=None, y_test_run=None, train_results=True):
        self.X_train_run = X_train_run
        self.X_test_run = X_test_run
        self.y_train_run = y_train_run
        self.y_test_run = y_test_run
        
        if train_results:
            y_train_perf = self.performance(X_train_run, y_train_run)
            y_train_auc = self.roc_auc_custom(X_train_run, y_train_run)
            y_test_perf = self.performance(X_test_run, y_test_run)
            y_test_auc = self.roc_auc_custom(X_test_run, y_test_run)
            train_matrix = self.class_matrix(X_train_run, y_train_run)
            test_matrix = self.class_matrix(X_test_run, y_test_run)
            return print(f"Train Report: \n{y_train_perf}\nROC-AUC Score:{y_train_auc}\n"), \
            print(f"Test Report: \n{y_test_perf}\nROC-AUC Score:{y_test_auc}")
        else:
            y_test_perf = self.performance(X_test_run, y_test_run)
            test_matrix = self.class_matrix(X_test_run, y_test_run)
            return print(f"Test Report: \n{y_test_perf}")
        
        
def record_results(model_name=None, model=None,
                   time_dic=None, time_results=None,
                    acc_dic=None, roc_auc_dic=None,
                  X_test=None, y_test=None):
    
    time_dic[model_name] = time_results
    acc_dic[model_name] = model.score(X_test, y_test)
    roc_auc_dic[model_name] = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])