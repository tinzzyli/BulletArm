import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--corrupt_func',type=str,default='gaussian_noise',help='The common corruption type used')
parser.add_argument('--severity', type=int, default=1, choices=[1,2,3,4,5],help='severity of corruptions')

args = parser.parse_args()

corrupt_func = args.corrupt_func
severity = args.severity