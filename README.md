# World Cup Predictor — Linear Algebra Team Rating System

A Julia project that rates World Cup teams attack and defense strength from match results, then simulates a full knockout tournament using linear algebra.

## Problem

Predicting match outcomes usually requires more than just win/loss records,it requires separating how good a teams attack is from how good its defense is, and accounting for the fact that goals scored depend on both teams involved in a match. This project builds a rating system that solves for those two numbers per team, directly from real match score data, using a system of linear equations.

## Approach
Model each match as data points. Every match produces two observations, the home team's scoring performance and the away team's scoring performance. Each expressed as a function of one team's attack rating and the opposing team's defense rating.
Build a sparse design matrix. Each team gets two unknowns (an attack rating and a defense rating), and each match contributes rows to a large sparse matrix linking those unknowns to observed (log transformed) goal counts.
Center the target values by subtracting the global average goals-per-match, isolating each team's rating relative to a shared baseline.
Solve the system via Normal Equations, adding ridge regularization (a small penalty term) to keep the system well-behaved since it's otherwise underdetermined with more unknowns than independent constraints.
Solve exactly using LUP (LU with partial pivoting) decomposition, rather than an iterative or approximate method.
Extract attack and defense ratings per team from the solved coefficients, and compute a combined "power rating" (attack minus defense) to rank teams.
Simulate the tournament, seeding the top 16 teams by power rating and running them through Round of 16 -> Quarterfinals -> Semifinals -> Final, with simulated matches based on each team's expected goals against their opponent (including penalty shootout style tiebreaking based on fractional expected goal advantage).
## Tools Used
Julia
LinearAlgebra (LUP decomposition, solving linear systems) 
SparseArrays (efficient representation of the match design matrix)
Printf (formatted output)
## How to Run It
Make sure Julia is installed along with the standard LinearAlgebra, SparseArrays, and Printf packages (all part of Julia's standard library).
Run the script — it will print the solved attack/defense ratings for every team, then simulate the full knockout bracket round by round, ending in a predicted champion.
## What This Demonstrates

This project applies core linear algebra concepts such as sparse matrix construction, least squares/normal equations, and LU decomposition to a real, interpretable problem: rating sports teams from raw match outcomes rather than relying on subjective rankings.
