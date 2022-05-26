# Artificial Economic Life
## Model of a simple stock market

[![version](https://img.shields.io/badge/version-0.0.1-yellow.svg)](https://semver.org)

This is a reproduction of a 1994 paper [Artificial economic life](https://deepblue.lib.umich.edu/handle/2027.42/31402) by Palmer, et al. It tries to implement ideas described in the paper, but is quite quick thanks to being implemented on a GPU.

<img
  src="https://i.imgur.com/iIKBljg.png"
  alt="Sample"
  title="Output of a simulation"
  style="display: inline-block; margin: 0 auto; max-width: 200px"
  width="700" height="700"/>
  
<img
  src="https://i.imgur.com/NaW5P8b.png"
  alt="Sample2"
  title="Analysis of strategies used"
  style="display: inline-block; margin: 0 auto; max-width: 200px"/>


## Dependencies
The only non-standard library is currently cupy and is not compatible with CPU at the moment.
