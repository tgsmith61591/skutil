from __future__ import division, absolute_import, print_function
import pandas as pd
import numpy as np
import abc


__all__ = [
	'ActStatisticalReport'
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


class ActStatisticalReport(object):
	"""A class that computes actuarial statistics
	for predictions given exposure and loss data.
	Primarily intended for use with 
	skutil.h2o.H2OActuarialRandomizedSearchCV

	Parameters
	----------
	n_groups : int, optional (default=10)
		The number of groups to use for computations.

	score_by : str, optional (default='lift')
		The metric to return for the _score method.
	"""


	__metrics__ = [
		'lift', 'gini'
	]

	__signs__ = {
		'lift' : -1,
		'gini' : -1
	}

	def __init__(self, n_groups=10, score_by='lift'):
		self.n_groups = 10
		self.score_by = score_by
		self.stats = {m:[] for m in self.__metrics__}

		# validate score_by
		if not score_by in self.__metrics__:
			raise ValueError('score_by must be in %s, but got %s'
				% (', '.join(self.__metrics__), score_by))

	def as_data_frame(self):
		"""Get the report in the form of a dataframe"""
		return pd.DataFrame.from_dict(self.stats)

	def _compute_stats(self, pred, expo, loss, prem):
		n_samples, n_groups = pred.shape[0], self.n_groups
		pred_ser = pd.Series(pred)
		loss_to_returns = np.sum(loss) / np.sum(prem)

		rank = pd.qcut(pred_ser, n_groups, labels=False)
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
		return self.stats[self.score_by][-1] * self.__signs__[self.score_by]


	def fit_fold(self, pred, expo, loss, prem=None):
		pred, expo, loss = _as_numpy(pred, expo, loss)

		if prem is None:
			prem = np.copy(expo)
		else:
			prem = _as_numpy(prem)

		# compute the stats
		tab, stats, n_groups = self._compute_stats(pred, expo, loss, prem)
		kwargs = {
			'pred' : pred,
			'expo' : expo,
			'loss' : loss,
			'prem' : prem,
			'stats': stats,
			'tab'  : tab,
			'n_groups' : n_groups
		}

		# compute the metrics
		for metric in self.__metrics__:
			self.stats[metric].append(
				getattr(self, '_%s'%metric)(**kwargs)
			)

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

