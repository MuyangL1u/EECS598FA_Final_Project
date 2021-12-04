# random_walk

random walk kernel of the https://github.com/talnish/iiswc21_rwalk repository.

1) Clone the repository:
`git clone https://github.com/arkhadem/random_walk.git`

2) CD to the repo directory:
`cd random_walk`

3) Build the executable code:
`./build`

4) Run the random walk kernel:
`sbatch run_test tgraph.wel`

You can make larger synthetic graphs using this script:

`python generate_synthetic.py -n #nodes# -e #edges# -s #seed#`
