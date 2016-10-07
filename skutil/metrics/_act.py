from __future__ import division, absolute_import, print_function
import pandas as pd
import numpy as np
import abc
from ..utils import safe_qcut


__all__ = [
	'GainsStatisticalReport'
]

def _as_numpy(*args):
	def _single_as_numpy(x):
		if not isinstance(x, np.ndarray):
			if hasattr(x, '__iter__'):
				return np.asarray(x)
			else:
				raise TypeError('cannot create numpy array out of type=%s' % type(x))
		else:
			return np.copy(x)

	arrs = [_single_as_numpy(x) for x in args]
	if len(arrs) == 1:
		arrs = arrs[0]

	return arrs


class GainsStatisticalReport(object):
	"""A class that computes actuarial statistics
	for predictions given exposure and loss data.
	Primarily intended for use with 
	skutil.h2o.H2OGainsRandomizedSearchCV

	Parameters
	----------
	n_groups : int, optional (default=10)
		The number of groups to use for computations.

	score_by : str, optional (default='lift')
		The metric to return for the _score method.

	n_folds : int, optional (default=None)
		The number of folds that are being fit. 
	"""

	# maximizing score functions must be multiplied by
	# -1 in order to most "minimize" some loss function
	_signs = {
		'lift' : -1,
		'gini' : -1
	}

	def __init__(self, n_groups=10, n_folds=None, n_iter=None, score_by='lift', iid=True):
		self.n_groups = 10
		self.score_by = score_by

		met = self._signs.keys()
		self.stats = {m:[] for m in met}
		self.sample_sizes = []

		self.n_folds = n_folds
		self.n_iter = n_iter
		self.iid = iid

		# validate score_by
		if not score_by in self._signs:
			raise ValueError('score_by must be in %s, but got %s'
				% (', '.join(met), score_by))

		if n_folds and not n_iter:
			raise ValueError('if n_folds is set, must set n_iter')

	def as_data_frame(self):
		"""Get the report in the form of a dataframe"""
		if not self.n_folds:
			# if there were no folds, these are each individual scores
			return pd.DataFrame.from_dict(self.stats)

		else:
			# otherwise they are cross validation scores...
			# ensure divisibility...
			n_obs, n_folds, n_iter = len(self.stats[self._signs.keys()[0]]), self.n_folds, self.n_iter
			if not (n_folds * n_iter) == n_obs:
				raise ValueError('n_obs is not divisible by n_folds and n_iter')

			new_stats = {}
			for metric in self._signs.keys():
				new_stats['%s_mean'%metric] = [] # the mean scores
				new_stats['%s_std' %metric] = [] # the std scores
				new_stats['%s_min' %metric] = [] # the min scores
				new_stats['%s_max' %metric] = [] # the max scores
				idx = 0

				for _ in range(n_iter):
					fold_score = 0
					n_test_samples = 0
					all_fold_scores = []

					for fold in range(n_folds):
						this_score = self.stats[metric][idx]
						this_n_test_samples = self.sample_sizes[idx]
						all_fold_scores.append(this_score)

						if self.iid:
							this_score *= this_n_test_samples
							n_test_samples += this_n_test_samples
						fold_score += this_score
						idx+=1

					if self.iid:
						fold_score /= float(n_test_samples)
					else:
						fold_score /= float(n_folds)

					# append the mean score, and then the std of the scores for the folds
					new_stats['%s_mean'%metric].append(fold_score)
					new_stats['%s_std' %metric].append(np.std(all_fold_scores))
					new_stats['%s_min' %metric].append(np.min(all_fold_scores))
					new_stats['%s_max' %metric].append(np.max(all_fold_scores))

			df = pd.DataFrame.from_dict(new_stats)

			# let's order by names
			return df[sorted(df.columns.values)]


	def _compute_stats(self, pred, expo, loss, prem):
		n_samples, n_groups = pred.shape[0], self.n_groups
		pred_ser = pd.Series(pred)
		loss_to_returns = np.sum(loss) / np.sum(prem)

		rank = safe_qcut(pred_ser, n_groups, labels=False)
		n_groups = np.amax(rank) + 1
		groups = np.arange(n_groups)

		tab = pd.DataFrame({
				'rank' : rank,
				'pred' : pred,
				'prem' : prem,
				'loss' : loss,
				'expo' : expo
			})

		grouped = tab[['rank','pred','prem','loss','expo']].groupby('rank')
		agg_rlr = (grouped['loss'].agg(np.sum) / grouped['prem'].agg(np.sum)) / loss_to_returns

		return tab, agg_rlr, n_groups


	def _score(self, _, pred, **kwargs):
		## For scoring from gridsearch...
		expo, loss, prem = kwargs.get('expo'), kwargs.get('loss'), kwargs.get('prem', None)
		self.fit_fold(pred, expo, loss, prem)

		# return the score we want... grid search is MINIMIZING
		# so we need to return negative for maximizing metrics
		return self.stats[self.score_by][-1] * self._signs[self.score_by]


	def fit_fold(self, pred, expo, loss, prem=None):
		"""Used to fit a single fold of predicted values, 
		exposure and loss data.
		"""
		pred, expo, loss = _as_numpy(pred, expo, loss)

		if prem is None:
			prem = np.copy(expo)
		else:
			prem = _as_numpy(prem)

		# compute the stats
		tab, stats, n_groups = self._compute_stats(pred, expo, loss, prem)
		kwargs = {
			'pred' : pred, 'expo' : expo,
			'loss' : loss, 'prem' : prem,
			'stats': stats, 'tab'  : tab,
			'n_groups' : n_groups
		}

		# compute the metrics. This relies on the convention
		# that the computation method is the name of the metric
		# preceded by an underscore...
		for metric in self._signs.keys():
			self.stats[metric].append(
				getattr(self, '_%s'%metric)(**kwargs)
			)

		self.sample_sizes.append(pred.shape[0])
		return self

	def _lift(self, **kwargs):
		agg = kwargs.pop('stats')
		n_groups = kwargs.pop('n_groups')

		f, l = agg[0], agg[n_groups-1]
		lft = l/f if l>f else f/l
		return lft


	def _gini(self, **kwargs):
		# we want a copy, not the original
		tab = kwargs.pop('tab').copy()[['pred','loss','prem','expo']]
		tab['idx'] = tab.index
		tab = tab.sort_values(by=['pred','idx'], axis=0, inplace=False)

		cpct = {x:tab[x].cumsum()/tab[x].sum() for x in ('prem', 'loss')}
		diff_pct = cpct['prem'] - cpct['loss']

		return 2 * np.average(diff_pct, weights=tab['expo'])

