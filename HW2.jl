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

# ╔═╡ fa5603e5-7d9d-453a-9c96-7d6a77f12127
md"""

#### Seperate on Wool Type

"""

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
Observate that on average, **wool of type A breaks more often than wool of type B**.

"""

# ╔═╡ 24ebf4be-0b4e-4158-ba05-b71ddfec3c44
md"""
## Model 1: Fully Pooled

The Poisson distribution gives the probability of observing some $k \in \mathbb{N}$ events in a given period of time, assuming that events occur independently at a constant rate. According to [^1], it can be used to model the distribution of the number of defects in a piece of material. Since every break in the wool is caused independently with no memory of the previous breaks, we will use the Poisson distribution to model the number of warp breaks in a loom:

$λ_0 = 27$
$λ_1 ∼ Exponential(λ_0)$
$obs[i] ∼ Poisson(λ_1) \space \forall i$

Where $obs[i]$ is the number of breaks in the $i$'th loom.
"""

# ╔═╡ 9ff9b9b0-3438-438d-920f-efb32a27cbca
@model function warp_breaks(breaks)
	λ0 = 27
	λ1 ~ Exponential(λ0)
	breaks ~ product_distribution(fill(Poisson(λ1), length(breaks)))
end

# ╔═╡ 6c478f37-ecb9-4a54-a252-8bfc21251a24
md"""
#### Sampling the Prior
"""

# ╔═╡ 4d6cf4f7-961a-4fb9-8ea6-5babe84cafa7
chn_prior = sample(warp_breaks(warpbreaks_df.Breaks), Prior(), 10000)

# ╔═╡ d334e9d1-f55c-44ea-b3bc-5b7afb7df84c
plot(chn_prior)

# ╔═╡ 58675fd9-d1fb-4d09-9879-12b495fa154a
describe(chn_prior)

# ╔═╡ 520a5338-d39d-4a41-a133-f9257a6b312e
md"""
#### Sampling the Posterior
"""

# ╔═╡ 7b61939d-4fdd-4cf3-9396-bc669d79c69d
chn = sample(warp_breaks(warpbreaks_df.Breaks), NUTS(), MCMCThreads(), 10000, 4)

# ╔═╡ c1e5aa03-e300-477b-bd42-b4d0c14457b2
plot(chn)

# ╔═╡ c7b066a2-cc0d-4d85-b7f9-a279b3345ee4
md"""

We can see that the sampler converges (all 4 chains are in agreement).

##### Seperating Wool Type
Let us estimate $λ_1$ seperately for each wool type:
"""

# ╔═╡ c8d43349-488d-4cd3-934b-88d334127272
begin
	chn_A = sample(warp_breaks(warpbreaks_df[warpbreaks_df.Wool .== "A", :Breaks]), NUTS(), MCMCThreads(), 10000, 1)
	chn_B = sample(warp_breaks(warpbreaks_df[warpbreaks_df.Wool .== "B", :Breaks]), NUTS(), MCMCThreads(), 10000, 1);
end

# ╔═╡ ff0dbf9b-0135-4c68-883b-ac30283ff9c7
begin
	plot(chn_A, label="Wool A")
	plot!(chn_B, label="Wool B", legend=:topright)
end

# ╔═╡ c5963b29-f843-4e38-ab21-51b3d891a197
md"""
## Plotting Prior and Posterior
Let's now plot the prior, posterior, and mean observation of $λ_1$:
"""

# ╔═╡ 88f73f6d-0709-46fb-b7a6-a9898a2f044c
begin
	density(chn[:,:,1], lab="posterior", color=:red)  # A density plot of the 1st sampled chain
	vline!([mean(warpbreaks_df.Breaks)], linewidth = 2, color=:yellow, label="mean observation",)  # The mean observation
	density!(chn_prior, label="prior",  legend=:topright, color=:cyan)  # The prior
end

# ╔═╡ b2f5b368-2940-442d-936f-58aec32c889e
md"""
The fact that the prior is so flat compared to the posterior alludes to the fact that it is a rather uninformative prior.
"""

# ╔═╡ d4974a79-c017-42b9-85b5-b21d46fddf0b
md"""
#### Seperating Wool Type
"""

# ╔═╡ 82e92527-9e2d-4427-bd84-6676b69ef9be
begin
	density(chn_A, lab="posterior (wool A)", color=:pink)  # A density plot of the 1st sampled chain
	density!(chn_B, lab="posterior (wool B)", color=:red)  # A density plot of the 1st sampled chain
	density!(chn_prior, label="prior",  legend=:topright, color=:cyan)  # The prior
end

# ╔═╡ 81251a40-160a-4a5b-bd5c-ca83de92a190
md"""
When fed just one category of data, the inferences produce different results. The posteriors seperated on $A$ and $B$ reflect the same trend we saw before, in the data: wool of type $A$ is estimated to break more often than wool of type $B$.  
"""

# ╔═╡ fc4fccfa-00ae-431d-b5e7-ddb7e8cde96d
md"""
## Model 2: Hierarchical on Wool
We would like to account for wool tension $T \in \{L, M, H\}$. 


"""

# ╔═╡ 7bc43c2b-4a41-4f3b-b193-875c7f558ce5
md"""
#### The Model
"""

# ╔═╡ 5ff7c63f-b926-4c11-b219-dbc6948b9cd7
@model function warp_breaks_hier(breaks)
	λ0 = 27
	λ1 ~ Exponential(λ0)
	breaks ~ product_distribution(fill(Poisson(λ1), length(breaks)))
end

# ╔═╡ 9fd59715-6aaf-4786-b417-39f795811e52
md"""
## Model 3: Fully Seperate

"""

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
# ╟─fa5603e5-7d9d-453a-9c96-7d6a77f12127
# ╠═8912c50a-3b43-4f0f-91ad-5c2a10f9d1a1
# ╠═60951dfa-e531-4fcd-ac2f-eab1bd7ba90b
# ╟─c5d27891-067c-46b0-90b7-bcdae15e6b8c
# ╟─24ebf4be-0b4e-4158-ba05-b71ddfec3c44
# ╠═9ff9b9b0-3438-438d-920f-efb32a27cbca
# ╟─6c478f37-ecb9-4a54-a252-8bfc21251a24
# ╠═4d6cf4f7-961a-4fb9-8ea6-5babe84cafa7
# ╠═d334e9d1-f55c-44ea-b3bc-5b7afb7df84c
# ╠═58675fd9-d1fb-4d09-9879-12b495fa154a
# ╟─520a5338-d39d-4a41-a133-f9257a6b312e
# ╠═7b61939d-4fdd-4cf3-9396-bc669d79c69d
# ╠═c1e5aa03-e300-477b-bd42-b4d0c14457b2
# ╟─c7b066a2-cc0d-4d85-b7f9-a279b3345ee4
# ╠═c8d43349-488d-4cd3-934b-88d334127272
# ╠═ff0dbf9b-0135-4c68-883b-ac30283ff9c7
# ╟─c5963b29-f843-4e38-ab21-51b3d891a197
# ╠═88f73f6d-0709-46fb-b7a6-a9898a2f044c
# ╟─b2f5b368-2940-442d-936f-58aec32c889e
# ╟─d4974a79-c017-42b9-85b5-b21d46fddf0b
# ╠═82e92527-9e2d-4427-bd84-6676b69ef9be
# ╟─81251a40-160a-4a5b-bd5c-ca83de92a190
# ╠═fc4fccfa-00ae-431d-b5e7-ddb7e8cde96d
# ╟─7bc43c2b-4a41-4f3b-b193-875c7f558ce5
# ╠═5ff7c63f-b926-4c11-b219-dbc6948b9cd7
# ╠═9fd59715-6aaf-4786-b417-39f795811e52
# ╟─bc121ee0-30df-4542-b25f-7c6f51b8d6d2
