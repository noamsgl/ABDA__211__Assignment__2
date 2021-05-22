### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ f4f05182-38b6-4bfc-bcbb-86ceb63cecbb
using Distributions, Turing, StatsPlots, Random, RDatasets, CSV

# ╔═╡ cff83100-b955-11eb-2950-75483cd235df
md""" 
# Assignment 2: Hierarchical Models

Name        |       Id
------------|----------
Noam Siegel  |314475062

# Problem 1: Warp Breaks

We define three random variables:
$B$ for Breaks,
$W$ for Wool,
$T$ for Tension.

Here, we infer the probability distribution of $B$ on $\mathbb{N}$ for each of two wool types $(W \in \{A, B\})$ and three tension types $(T \in \{L, M, H\})$. We compare the results under separate and hierarchical models.
"""

# ╔═╡ 4b56e944-8dda-4fa9-a4e9-71d0255110aa
md"""
## Data
"""

# ╔═╡ 74905b50-46b0-41a1-86bb-5d151e3e3a58
begin
	warpbreaks_df = RDatasets.dataset("datasets", "warpbreaks")
	warpbreaks_df
end

# ╔═╡ 8e981ff6-e0a1-45f6-98ef-113432c13808
histogram(warpbreaks_df.Breaks, bins=20,xlabel="Count of Warp Breaks", ylabel="Count of Looms", label="All Data")

# ╔═╡ b44ae535-e44e-4dbc-a97e-bb50ca90e2bb
summarystats(warpbreaks_df.Breaks)

# ╔═╡ 24ebf4be-0b4e-4158-ba05-b71ddfec3c44
md"""
## Fully Pooled Model

The Poisson distribution gives the probability of observing some $k \in \mathbb{N}$ events in a given period of time, assuming that events occur independently at a constant rate. According to [^1], it can be used to model the distribution of the number of defects in a piece of material. Since every break in the wool is caused independently with no memory of the previous breaks, we will use the Poisson distribution to model the number of warp breaks in a loom:

$λ_0 = 27$
$λ_1 ∼ Exponential(λ_0)$
$obs[i] ∼ Poisson(λ_1) \space \forall i$

Where $obs[i]$ is the number of breaks in the $i$'th loom.
"""

# ╔═╡ 9ff9b9b0-3438-438d-920f-efb32a27cbca
@model function warp_breaks(obs; mean_break_rate=27)
	λ1 ~ Exponential(mean_break_rate)
	obs ~ product_distribution(fill(Poisson(λ1), length(obs)))
	return mean_break_rate
end

# ╔═╡ 6c478f37-ecb9-4a54-a252-8bfc21251a24
md"""
#### Sampling the Prior
"""

# ╔═╡ 520a5338-d39d-4a41-a133-f9257a6b312e
md"""
#### Sampling the posterior
"""

# ╔═╡ 7b61939d-4fdd-4cf3-9396-bc669d79c69d
begin
	observations = warpbreaks_df.Breaks
	chn = sample(warp_breaks(observations, mean_break_rate=27), NUTS(), MCMCThreads(), 10000, 4)
end

# ╔═╡ 4d6cf4f7-961a-4fb9-8ea6-5babe84cafa7
begin
	chn_prior = sample(warp_breaks(observations, mean_break_rate=27), Prior(), 10000)
end

# ╔═╡ d334e9d1-f55c-44ea-b3bc-5b7afb7df84c
plot(chn_prior)

# ╔═╡ c1e5aa03-e300-477b-bd42-b4d0c14457b2
plot(chn)

# ╔═╡ 8912c50a-3b43-4f0f-91ad-5c2a10f9d1a1
begin
	bins = 0:5:100
	xlim = (0,100)
	@df warpbreaks_df groupedhist(:Breaks, group=:Wool, bar_position = :dodge, bins=bins, ylabel="Loom Counts")
end

# ╔═╡ 60951dfa-e531-4fcd-ac2f-eab1bd7ba90b
summarystats(warpbreaks_df.Breaks), summarystats(warpbreaks_df[warpbreaks_df.Wool .== "A",:].Breaks), summarystats(warpbreaks_df[warpbreaks_df.Wool .== "B",:].Breaks)

# ╔═╡ c5d27891-067c-46b0-90b7-bcdae15e6b8c
md"""
Thus, wool A breaks more often than wool B.
We want to utilize this information to 

"""

# ╔═╡ 7bc43c2b-4a41-4f3b-b193-875c7f558ce5
md"""
#### The Model
"""

# ╔═╡ 7f3140f5-0cda-474a-9b5b-a68ec5e68b09
md"""
#### Samples
"""

# ╔═╡ 4f764ac8-8b2d-4870-8aaf-0c3b7c7ae133
function take(a=1, b=1)
	u = rand()
	if u >= 0.5
		x = a - b * log(2*(1-u))
	else
		x = a + b * log(2*u)
	end
	return x
end

# ╔═╡ edb063c6-4466-4b74-995a-043f438c3376
vec = zeros(4)

# ╔═╡ b84b36b0-6baf-4b56-84fd-c2d1d3e39e8e
take.(vec)

# ╔═╡ d9ed9d93-a252-4fe6-a541-0f8d5e571a7b
begin
	result = zeros(10000)
	result = take.(result)
	histogram(result)
end

# ╔═╡ bc121ee0-30df-4542-b25f-7c6f51b8d6d2
md"""


### References

[^1]: Krishnamoorthy, Kalimuthu. Handbook of statistical distributions with applications. CRC Press, 2016. (p. 90)

"""

# ╔═╡ Cell order:
# ╠═f4f05182-38b6-4bfc-bcbb-86ceb63cecbb
# ╠═cff83100-b955-11eb-2950-75483cd235df
# ╟─4b56e944-8dda-4fa9-a4e9-71d0255110aa
# ╠═74905b50-46b0-41a1-86bb-5d151e3e3a58
# ╠═8e981ff6-e0a1-45f6-98ef-113432c13808
# ╠═b44ae535-e44e-4dbc-a97e-bb50ca90e2bb
# ╟─24ebf4be-0b4e-4158-ba05-b71ddfec3c44
# ╠═9ff9b9b0-3438-438d-920f-efb32a27cbca
# ╟─6c478f37-ecb9-4a54-a252-8bfc21251a24
# ╠═4d6cf4f7-961a-4fb9-8ea6-5babe84cafa7
# ╠═d334e9d1-f55c-44ea-b3bc-5b7afb7df84c
# ╟─520a5338-d39d-4a41-a133-f9257a6b312e
# ╠═7b61939d-4fdd-4cf3-9396-bc669d79c69d
# ╠═c1e5aa03-e300-477b-bd42-b4d0c14457b2
# ╠═8912c50a-3b43-4f0f-91ad-5c2a10f9d1a1
# ╠═60951dfa-e531-4fcd-ac2f-eab1bd7ba90b
# ╠═c5d27891-067c-46b0-90b7-bcdae15e6b8c
# ╟─7bc43c2b-4a41-4f3b-b193-875c7f558ce5
# ╟─7f3140f5-0cda-474a-9b5b-a68ec5e68b09
# ╠═4f764ac8-8b2d-4870-8aaf-0c3b7c7ae133
# ╠═edb063c6-4466-4b74-995a-043f438c3376
# ╠═b84b36b0-6baf-4b56-84fd-c2d1d3e39e8e
# ╠═d9ed9d93-a252-4fe6-a541-0f8d5e571a7b
# ╟─bc121ee0-30df-4542-b25f-7c6f51b8d6d2
