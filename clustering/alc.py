import numpy as np
from numba import jit

class ALCAdapter:
    """Project: “Agglomerative Likelihood Clustering”
    Author : Lionel Yelibi, 2019, University of Cape Town.
    Copyright SPC, 2019, 2020, 2021
    Potts Model Clustering.
    Agglomerative Likelihood Clustering
    See pre-print: https://arxiv.org/abs/1908.00951
    GNU GPL
    This file is part of ALC
    ALC is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    ALC is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>."""


    ''' the cluster function:
        compute the likelihood which occurs when two objects are clustered or
        the object own likelihood. By design if a cluster only containts one object
        its likelihood will be 0'''


    def clus_lc(self, gij, gii, gjj, ns=2):
        """ variables description:
            ns is the size of the cluster.
            cs is the intracluster correlation.
            gij, gii, and gjj respectively are
            correlations relating to interactions between objects i and j, and
            their respective self-correlations. self-correlations (i.e. gii, gjj)
            are 1 for individual objects and >1 for clusters"""

        if ns == 1:
            return 0
        ''' intracluster correlation'''
        cs = 2 * gij + gii + gjj - 1e-3
        ''' relatively low cs means noisy and suboptimal clusters.
        The coupling parameter gs (see paper) isn't not defined'''
        if cs <= ns:
            return 0
        return 0.5 * (np.log(ns / cs) + (ns - 1) * np.log((ns ** 2 - ns) / (ns ** 2 - cs)))


    ''' aspc only requires a correlation matrix as input:
        here we convert the correlation to a dictionary for convenience. adding new
        entries in a dict() is much faster than editing a numpy matrix'''

    def fit_predict(self, X, cn=None):
        G = np.corrcoef(X)
        N = len(G)
        # build correlation dictionary
        gdic = dict(enumerate(dict(enumerate(row)) for row in G))
        clike = {i: 0 for i in range(N)}
        del G

        # tracker holds the current clusters (list of indices for each cluster)
        tracker = {i: [i] for i in range(N)}

        # start with all points unclustered
        other_keys = list(range(N))

        # continue until the desired number of clusters is reached
        while len(tracker) != cn:
            # keep other_keys in sync with tracker – drop any stale labels
            other_keys = [k for k in other_keys if k in tracker]
            if not other_keys:
                break

            # choose a node at random from the current active labels
            if len(other_keys) > 1:
                node = np.random.choice(other_keys)
            else:
                # when all points have been considered, choose from tracker directly
                if len(tracker) != 1:
                    node = np.random.choice(list(tracker.keys()))
                else:
                    cn = 1
                    break

            # build a fresh neighbour list from tracker keys
            nbor = [k for k in tracker.keys() if k != node]
            costs = np.zeros(len(nbor))
            indices = [(node, key) for key in nbor]
            node_lc = self.clus_lc(0, gdic[node][node], 0, ns=len(tracker[node]))

            k = 0
            for i, j in indices:
                costs[k] = self.clus_lc(
                    gdic[i][j], gdic[i][i], gdic[j][j],
                    ns=len(tracker[i]) + len(tracker[j])
                ) - (node_lc + self.clus_lc(0, gdic[j][j], 0, ns=len(tracker[j])))
                k += 1

            next_merge = np.argmax(costs)

            # stopping conditions
            if costs[next_merge] <= 0:
                if len(other_keys) > 1:
                    # remove this node from further consideration (but only if present)
                    if node in other_keys:
                        other_keys.remove(node)
                    continue
                elif not cn:
                    cn = len(tracker)
                    break
                else:
                    pass

            # determine the two clusters to be merged
            label_a = node
            label_b = indices[next_merge][1]
            new_label = list(tracker.keys())[-1] + 1

            clike[new_label] = self.clus_lc(
                gdic[label_a][label_b],
                gdic[label_a][label_a],
                gdic[label_b][label_b],
                ns=len(tracker[label_a]) + len(tracker[label_b])
            )
            # remove old entries from clike
            clike.pop(label_a, None)
            clike.pop(label_b, None)

            # only update other_keys when a positive cost is found
            if costs[next_merge] > 0:
                other_keys = [k for k in tracker.keys() if k not in (label_a, label_b)]
                other_keys.append(new_label)

            # update tracker, gdic and remove merged clusters
            # remove label_b from nbor safely
            if label_b in nbor:
                nbor.remove(label_b)
            tracker[new_label] = tracker[label_a] + tracker[label_b]
            gdic[new_label] = {}
            gdic[new_label][new_label] = (
                    2 * gdic[label_a][label_b]
                    + gdic[label_a][label_a]
                    + gdic[label_b][label_b]
            )

            for key in nbor:
                gdic[new_label][key] = gdic[label_a][key] + gdic[label_b][key]
                gdic[key][new_label] = gdic[label_a][key] + gdic[label_b][key]

            # delete old labels from tracker
            tracker.pop(label_a, None)
            tracker.pop(label_b, None)

        # build the final solution vector
        solution = np.zeros(N, dtype=int)
        k = 1
        for cluster in tracker.keys():
            cluster_members = tracker[cluster]
            solution[cluster_members] = k
            k += 1
        return solution












