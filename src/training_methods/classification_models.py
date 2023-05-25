from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, RandomTreesEmbedding, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier

#for metrics
from sklearn.metrics import accuracy_score

class classification_models(object):

    #the dataset is already processed
    def __init__(
            self, 
            dataset=None, 
            response=None,
            apply_cv=False,
            cv_value=None,
            test_size=0.3):

        self.dataset = dataset
        self.response = response

        self.apply_cv = apply_cv
        self.cv_value = cv_value
        self.test_size = test_size

        self.clf_model = None

    def prepare_dataset(
            self,
            random_state=None,
            shuffle=True):
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.dataset, 
            self.response, 
            test_size=self.test_size, 
            random_state=random_state,
            shuffle=shuffle)
        
    def apply_svc(
            self,
            C=1.0,
            kernel='rbf',
            degree=3,
            gamma='scale',
            coef0=0.0,
            shrinking=True,
            probability=False,
            tol=1e-3,
            cache_size=200,
            class_weight=None,
            verbose=False,
            max_iter=-1,
            decision_function_shape='ovr',
            break_ties=False,
            random_state=None):
        
        #instance
        self.clf_model =SVC(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state
        )
    
    def apply_NuSVC(
            self,
            nu=0.5,
            kernel='rbf',
            degree=3,
            gamma='scale',
            coef0=0.0,
            shrinking=True,
            probability=False,
            tol=1e-3,
            cache_size=200,
            class_weight=None,
            verbose=False,
            max_iter=-1,
            decision_function_shape='ovr',
            break_ties=False,
            random_state=None
            ):
        
        #instance
        self.clf_model =NuSVC(
            nu=nu,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state
        )

    def apply_LinearSVC(
            self,
            penalty='l2',
            loss='squared_hinge',
            dual=True,
            tol=1e-4,
            C=1.0,
            multi_class='over',
            fit_intercept=True,
            intercept_scaling=1.0,
            class_weight=None,
            verbose=0,
            random_state=None,
            max_iter=1000):

        self.clf_model = LinearSVC(
            penalty=penalty,
            loss=loss,
            dual=dual,
            tol=tol,
            C=C,
            multi_class=multi_class,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            verbose=verbose,
            random_state=random_state,
            max_iter=max_iter)
        
    def apply_SGDClassifier(
            self,
            loss="hinge",
            penalty="l2",
            alpha=0.0001,
            l1_ratio=0.15,
            fit_intercept=True,
            max_iter=1000,
            tol=1e-3,
            shuffle=True,
            verbose=0,
            epsilon=0.1,
            n_jobs=-1,
            random_state=None,
            learning_rate='optimal',
            eta0=0.0,
            power_t=0.5,
            early_stopping=False,
            validation_fraction=0.1,
            n_iter_no_change=5,
            class_weight=None,
            warm_start=False,
            average=False):
        
        self.clf_model = SGDClassifier(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            shuffle=shuffle,
            verbose=verbose,
            epsilon=epsilon,
            n_jobs=n_jobs,
            random_state=random_state,
            learning_rate=learning_rate,
            eta0=eta0,
            power_t=power_t,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            class_weight=class_weight,
            warm_start=warm_start,
            average=average
        )

    def apply_KNeighborsClassifier(
            self,
            n_neighbors=5,
            weights='uniform',
            algorithm='auto',
            leaf_size=30,
            p=2,
            metric='minkowski',
            metric_params=None,
            n_jobs=-1):

        self.clf_model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs
        )        

    def apply_GaussianProcessClassifier(
            self,
            kernel=None,
            optimizer=None,
            n_restarts_optimizer=0,
            max_iter_predict=100,
            warm_start=False,
            copy_X_train=False,
            random_state=None,
            multi_class='one_vs_rest',
            n_jobs=-1):
        
        self.clf_model = GaussianProcessClassifier(
            kernel=kernel,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            max_iter_predict=max_iter_predict,
            warm_start=warm_start,
            copy_X_train=copy_X_train,
            random_state=random_state,
            multi_class=multi_class,
            n_jobs=n_jobs
        )

    def apply_GaussianNB(
            self,
            priors=None,
            var_smoothing=1e-9):
        
        self.clf_model = GaussianNB(
            priors=priors,
            var_smoothing=var_smoothing
        )

    def apply_MultinomialNB(
            self,
            alpha=1.0,
            force_alpha=True,
            fit_prior=True,
            class_prior=None):
        
        self.clf_model = MultinomialNB(
            alpha=alpha,
            force_alpha=force_alpha,
            fit_prior=fit_prior,
            class_prior=class_prior
        )

    def apply_ComplementNB(
            self,
            alpha=1.0,
            force_alpha=True,
            fit_prior=True,
            class_prior=None,
            norm=False):
        
        self.clf_model = ComplementNB(
            alpha=alpha,
            force_alpha=force_alpha,
            fit_prior=fit_prior,
            class_prior=class_prior,
            norm=norm
        )

    def apply_BernoulliNB(
            self,
            alpha=1.0,
            force_alpha=True,
            binarize=0.0,
            fit_prior=True,
            class_prior=None):
        
        self.clf_model = BernoulliNB(
            alpha=alpha,
            force_alpha=force_alpha,
            binarize=binarize,
            fit_prior=fit_prior,
            class_prior=class_prior
        )

    def apply_CategoricalNB(
            self,
            alpha=1.0,
            force_alpha=True,
            fit_prior=True,
            class_prior=None,
            min_categories=None):
        
        self.clf_model = CategoricalNB(
            alpha=alpha,
            force_alpha=force_alpha,
            fit_prior=fit_prior,
            class_prior=class_prior,
            min_categories=min_categories
        )

    def apply_DecisionTreeClassifier(
            self,
            criterion='gini',
            splitter='best',
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=None,
            random_state=None,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            class_weight=None,
            ccp_alpha=0.0):
        
        self.clf_model = DecisionTreeClassifier(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha
        )

    def apply_BaggingClassifier(
            self,
            estimator=None,
            n_estimators=10,
            max_samples=1.0,
            max_features=1.0,
            bootstrap=True,
            bootstrap_features=False,
            oob_score=False,
            warm_start=False,
            n_jobs=-1,
            random_state=None,
            verbose=0):
        
        self.clf_model = BaggingClassifier(
            estimator=estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose
        )

    def apply_RandomForestClassifier(
            self,
            n_estimators=100,
            criterion='gini',
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features='sqrt',
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=-1,
            random_state=None,
            verbose=0,
            warm_start=False,
            class_weight=None,
            ccp_alpha=0.0,
            max_samples=None):
        
        self.clf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples
        )

    def apply_ExtraTreesClassifier(
            self,
            n_estimators=100,
            criterion='gini',
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features='sqrt',
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=False,
            oob_score=False,
            n_jobs=-1,
            random_state=None,
            verbose=0,
            warm_start=False,
            class_weight=None,
            ccp_alpha=0.0,
            max_samples=None):

        self.clf_model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples
        )

    def apply_RandomTreesEmbedding(
            self,
            n_estimators=100,
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            sparse_output=True,
            n_jobs=-1,
            random_state=None,
            verbose=0,
            warm_start=False):

        self.clf_model = RandomTreesEmbedding(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            sparse_output=sparse_output,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start
        )

    def apply_AdaBoostClassifier(
            self,
            estimator=None,
            n_estimators=50,
            learning_rate=1.0,
            algorithm='SAMME.R',
            random_state=None):
        
        self.clf_model = AdaBoostClassifier(
            estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            random_state=random_state
        )
    
    def apply_GradientBoostingClassifier(
            self,
            loss='log_loss',
            learning_rate=0.1,
            n_estimators=100,
            subsample=1.0,
            criterion='friedman_mse',
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_depth=3,
            min_impurity_decrease=0.0,
            init=None,
            random_state=None,
            max_features=None,
            verbose=0,
            max_leaf_nodes=None,
            warm_start=False,
            validation_fraction=0.1,
            n_iter_no_change=None,
            tol=1e-4,
            ccp_alpha=0.0):
        
        self.clf_model = GradientBoostingClassifier(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            criterion=criterion,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            init=init,
            random_state=random_state,
            max_features=max_features,
            verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            ccp_alpha=ccp_alpha
        )

    def apply_HistGradientBoostingClassifier(
            self,
            loss='log_loss',
            learning_rate=0.1,
            max_iter=100,
            max_leaf_nodes=31,
            max_depth=None,
            min_samples_leaf=20,
            l2_regularization=0.0,
            max_bins=255,
            categorical_features=None,
            monotonic_cst=None,
            interaction_cst=None,
            warm_start=False,
            early_stopping='auto',
            scoring='loss',
            validation_fraction=0.1,
            n_iter_no_change=10,
            tol=1e-7,
            verbose=0,
            random_state=None,
            class_weight=None):

        self.clf_model = HistGradientBoostingClassifier(
            loss=loss,
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            max_bins=max_bins,
            categorical_features=categorical_features,
            monotonic_cst=monotonic_cst,
            interaction_cst=interaction_cst,
            warm_start=warm_start,
            early_stopping=early_stopping,
            scoring=scoring,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            class_weight=class_weight
        )
    
    def apply_VotingClassifier(
            self,
            estimators=[('dt', DecisionTreeClassifier()), ('bg', BaggingClassifier())],
            voting='hard',
            weights=None,
            n_jobs=-1,
            flatten_transform=True,
            verbose=False):
        
        self.clf_model = VotingClassifier(
            estimators=estimators,
            voting=voting,
            weights=weights,
            n_jobs=n_jobs,
            flatten_transform=flatten_transform,
            verbose=verbose
        )
        
    def apply_MLPClassifier(
            self,
            hidden_layer_sizes=(100,),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='constant',
            learning_rate_init=0.001,
            power_t=0.5,
            max_iter=200,
            shuffle=True,
            random_state=None,
            tol=1e-4,
            verbose=False,
            warm_start=False,
            momentum=0.9,
            nesterovs_momentum=True,
            early_stopping=False,
            validation_fraction=0.1,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            n_iter_no_change=10,
            max_fun=15000):
        
        self.clf_model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            max_fun=max_fun
        )

    def __get_accuracy(
            self, 
            predictions=None,
            real_valeus=None,
            normalize=True,
            sample_weight=None):
        
        return None
    
    def __get_metrics(self):

        predictions = self.clf_model.predict(self.X_test)

    def training_process(self):

        if self.clf_model:
            self.clf_model.fit(self.X_train, self.y_train)
        else:
            return "You need to instance a model"      

        



    

