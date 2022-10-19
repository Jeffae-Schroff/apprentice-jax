from paramtune import Paramtune
import argparse
parser = argparse.ArgumentParser(description="Outer Loop Optimimzation")
add_arg = parser.add_argument
add_arg("target_data", help="for now this is a json")
add_arg("initial_guess", help="list of length equal to number of parameters")
add_arg("p_coeffs_npz", help="")
add_arg("chi2res_npz", help="")
add_arg("-cov_npz", help="")
args = parser.parse_args()

initial_guess_list = [float(s) for s in args.initial_guess.split(',')]
p = Paramtune(args.target_data, initial_guess_list, args.p_coeffs_npz, args.chi2res_npz, cov_npz = args.cov_npz)