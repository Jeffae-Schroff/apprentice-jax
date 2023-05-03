import matplotlib.pyplot as plt
import jax.numpy as jnp

def graph_envelope_target(self, observable = None, save_folder = None):
    place = 0
    if observable:
        obs_names = [observable]
    else:
        obs_names = self.fits.obs_index.keys()

    for obs_name in obs_names:
        target_bins = len(jnp.intersect1d(self.target_binidns, jnp.array(self.fits.index[obs_name])))
        if target_bins > 0:
            self.fits.graph_envelope([obs_name])
            plt.stairs(self.target_values[place:place+target_bins], range(target_bins + 1), label = 'target')
            plt.legend()
            place += target_bins
            if(save_folder):
                plt.savefig(save_folder + "/" + obs_name + ".pdf")
def graph_envelope(self, graph_obs = None):
    ymin = self.Y.min(axis=1)
    ymax = self.Y.max(axis=1)
    if graph_obs is None:
        graph_obs = self.obs_index.keys()
    elif type(graph_obs) == str:
        graph_obs = [graph_obs]
    for obs in graph_obs:
        obs_bin_idns = jnp.array(self.index[obs])
        plt.figure()
        plt.title("Envelope of " + obs)
        #Might be something like "number of events", but depends on what observable is.
        plt.ylabel("Placeholder")
        plt.xlabel(obs + " bins")
        num_bins = len(obs_bin_idns)
        num_ticks = 7 if num_bins > 14 else num_bins #make whole numbers
        edges = range(num_bins + 1)
        plt.xticks([round(x/num_ticks) for x in range(0, num_bins*num_ticks, num_bins)]+[num_bins])
        plt.stairs(jnp.take(ymin, obs_bin_idns), edges, label = 'min')
        plt.stairs(jnp.take(ymax, obs_bin_idns), edges, label = 'max')
        plt.legend()
