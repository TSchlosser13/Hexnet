'''****************************************************************************
 * models_sklearn.py: scikit-learn Models
 ******************************************************************************
 * v0.1 - 01.03.2019
 *
 * Copyright (c) 2019 Tobias Schlosser (tobias@tobias-schlosser.net)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 ****************************************************************************'''


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble              import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model          import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes           import GaussianNB
from sklearn.neighbors             import KNeighborsClassifier
from sklearn.neural_network        import MLPClassifier
from sklearn.svm                   import LinearSVC, SVC
from sklearn.tree                  import DecisionTreeClassifier


def model_sklearn_LinearDiscriminantAnalysis(**kwargs):
	return LinearDiscriminantAnalysis(**kwargs)

def model_sklearn_QuadraticDiscriminantAnalysis(**kwargs):
	return QuadraticDiscriminantAnalysis(**kwargs)

def model_sklearn_ExtraTreesClassifier(n_jobs=-1, verbose=1, **kwargs):
	return ExtraTreesClassifier(n_jobs=n_jobs, verbose=verbose, **kwargs)

def model_sklearn_RandomForestClassifier(n_jobs=-1, verbose=1, **kwargs):
	return RandomForestClassifier(n_jobs=n_jobs, verbose=verbose, **kwargs)

def model_sklearn_LogisticRegression(n_jobs=-1, verbose=1, **kwargs):
	return LogisticRegression(n_jobs=n_jobs, verbose=verbose, **kwargs)

# TODO: add predict_proba support
def model_sklearn_RidgeClassifier(**kwargs):
	return RidgeClassifier(**kwargs)

def model_sklearn_GaussianNB(**kwargs):
	return GaussianNB(**kwargs)

def model_sklearn_KNeighborsClassifier(n_jobs=-1, **kwargs):
	return KNeighborsClassifier(n_jobs=n_jobs, **kwargs)

def model_sklearn_MLPClassifier(verbose=1, **kwargs):
	return MLPClassifier(verbose=verbose, **kwargs)

def model_sklearn_LinearSVC(verbose=1, **kwargs):
	# TODO: add predict_proba support
	# return LinearSVC(verbose=verbose, **kwargs)
	return SVC(kernel='linear', probability=True, verbose=verbose, **kwargs)

def model_sklearn_SVC(verbose=1, **kwargs):
	return SVC(kernel='rbf', probability=True, verbose=verbose, **kwargs)

def model_sklearn_DecisionTreeClassifier(**kwargs):
	return DecisionTreeClassifier(**kwargs)


