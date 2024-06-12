
import numpy as np
import matplotlib.pyplot as plt


class GraphDAG(object):

    def __init__(self, est_dag, true_dag=None, show=True, save_name=None):

        self.est_dag = est_dag
        self.true_dag = true_dag
        self.show = show
        self.save_name = save_name

        if not show and save_name is None:
            raise ValueError('Neither display nor save the picture! ' + \
                             'Please modify the parameter show or save_name.')

        GraphDAG._plot_dag(self.est_dag, self.true_dag, self.show, self.save_name)
    #draw matrix of result
    @staticmethod
    def _plot_dag(est_dag, true_dag, show=True, save_name=None):

        if isinstance(true_dag, np.ndarray):

            for i in range(len(true_dag)):
                if est_dag[i][i] == 1:
                    est_dag[i][i] = 0
                if true_dag[i][i] == 1:
                    true_dag[i][i] = 0

            fig, (ax1, ax2) = plt.subplots(figsize=(8, 3), ncols=2)

            ax1.set_title('est_graph')
            map1 = ax1.imshow(est_dag, cmap='Greys', interpolation='none')
            fig.colorbar(map1, ax=ax1)

            ax2.set_title('true_graph')
            map2 = ax2.imshow(true_dag, cmap='Greys', interpolation='none')
            fig.colorbar(map2, ax=ax2)
            
            if save_name is not None:
                fig.savefig(save_name)
            if show:
                plt.show()

        elif isinstance(est_dag, np.ndarray):

            for i in range(len(est_dag)):
                if est_dag[i][i] == 1:
                    est_dag[i][i] = 0

            fig, ax1 = plt.subplots(figsize=(4, 3), ncols=1)

            ax1.set_title('est_graph')
            map1 = ax1.imshow(est_dag, cmap='Greys', interpolation='none')
            fig.colorbar(map1, ax=ax1)

            if save_name is not None:
                fig.savefig(save_name)
            if show:
                plt.show()

        else:
            raise TypeError("Input data is not numpy.ndarray!")
