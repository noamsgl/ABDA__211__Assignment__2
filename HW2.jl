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
warpbreaks_df = RDatasets.dataset("datasets", "warpbreaks")

# ╔═╡ 8e981ff6-e0a1-45f6-98ef-113432c13808
histogram(warpbreaks_df.Breaks, bins=20,xlabel="Count of Warp Breaks", ylabel="Count of Looms", label="All Data")

# ╔═╡ b44ae535-e44e-4dbc-a97e-bb50ca90e2bb
summarystats(warpbreaks_df.Breaks), std(warpbreaks_df.Breaks)

# ╔═╡ fa5603e5-7d9d-453a-9c96-7d6a77f12127
md"""

#### Seperate on Wool Type

"""

# ╔═╡ 8912c50a-3b43-4f0f-91ad-5c2a10f9d1a1
let
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

# ╔═╡ 9fd59715-6aaf-4786-b417-39f795811e52
md"""
## Model 2: Fully Seperate
Here, we let each experiment have it's own set of parameters (not shared).

"""

# ╔═╡ 70c61890-a65f-49a8-be91-9bb9ddbcbf00
@model function warp_breaks_seperate(breaks, tension)
	λ0 = 27
	λ1 ~ product_distribution(fill(Exponential(λ0), length(breaks)))
	for i in eachindex(breaks)
		breaks[i] ~ Poisson(λ1[i])
	end
end

# ╔═╡ c53f412d-d763-4def-838b-517b5c4e9d81
chn_seperate = sample(warp_breaks_seperate(warpbreaks_df.Breaks, warpbreaks_df.Tension), NUTS(), MCMCThreads(), 10000, 4)

# ╔═╡ c89b51c7-e6ff-4cf7-9ab2-0b536784b686
plot(chn_seperate)

# ╔═╡ a63016f7-1954-4434-bc3f-bc8cb844057e
inferred = mean(Array(group(chn_seperate, :λ1)), dims = 1)'

# ╔═╡ 64fd1f37-63f0-45ee-be3b-514b1193c7cb
md"""
We can now plot the posterior for each mean break rate, compared to the observed break rate, for the three types of wool.
"""

# ╔═╡ d5a0f661-41e6-4e02-b3ad-dd73ce1f30fc
let
	p = plot(layout = (1,3), size = (1000,300))
	for (i, t) in enumerate(unique(warpbreaks_df.Tension))
		indices = warpbreaks_df.Tension .== t
		scatter!(warpbreaks_df.Breaks[indices], inferred[indices],
			xlabel="Observed Breaks", ylabel="Inferred Mean Break Rate", title="Tension $t", label=:Wool, subplot=i)
		plot!([0, 80], [0, 80], subplot=i, line=(1, :dash, :green), label=missing)
	end
	p
end

# ╔═╡ 8cbd6987-c9a0-471e-85ad-e987e7383b12
unique(warpbreaks_df.Tension)

# ╔═╡ b31e0f83-2f64-44c6-998c-fd7be90cedd5
histogram(warpbreaks_df.Breaks, bins=30)

# ╔═╡ fc4fccfa-00ae-431d-b5e7-ddb7e8cde96d
md"""
## Model 2: Hierarchical on Wool
We would like to account for wool tension $T \in \{L, M, H\}$ in a multilevel model.

$λ_0 = 27$
$λ_1 ∼ Exponential(λ_0)$
$obs[i] ∼ Poisson(λ_1) \space \forall i$

Where $obs[i]$ is the number of breaks in the $i$'th loom.

"""

# ╔═╡ 7bc43c2b-4a41-4f3b-b193-875c7f558ce5
md"""
#### The Model
"""

# ╔═╡ 5ff7c63f-b926-4c11-b219-dbc6948b9cd7
# @model function warp_breaks_hier(breaks, tension)
# 	λ0 = 27
# 	σ ~ Exponential(1)  # Prior for std of tensions
	
# 	α ~ MvNormal(fill(λ0, length(tension), 1.5)  # Prior for average tension break rate
# 	MvNormal(fill(μ0, length(tension), σ)
# 	λ1 ~ Exponential(λ0)
# 	breaks ~ product_distribution(fill(Poisson(λ1), length(breaks)))
# end

# ╔═╡ 78586ab6-e1c1-4731-aa2b-6bd73c1d6d64
md"""
## Problem 2: Norfolk City Salaries
"""

# ╔═╡ b99d128a-6179-4644-8bff-8d7b23cc7a5e
md"""
### Exploratory Data Analysis

First, we will explore the dataset.
"""

# ╔═╡ 197973a1-2cde-4be5-b4dc-608ec55f57f4
md"""
Thus, the original dataset includes 4399 employees and 7 fields of information.
Let's trim the whitespaces surrounding "Department  " and filter unnecessary columns.
"""

# ╔═╡ 00dae8e7-1358-4e49-a851-18812969199e
md"""### Exploring Base Salary

Let us plot a histogram of the base salary.
"""

# ╔═╡ 5c538050-fe6d-48bc-b206-e3bd45462cfe
md"""
<b>Observation</b>: The salary distribution is **clustered**: employees either have salaries in the range (9,150) or in (3500, 260000).

**Explanation**: The base salaries reflect either an hourly rate or a monthly income.

**Action**: In order to correct for the different values in this column, we will decide that any value under 1000 is an hourly rate and any value at or above 1000 is a monthly salary. We will summarize this in a new column "Monthly Salary".

This is the histogram of the new column:"""

# ╔═╡ 4c671e0d-6020-4a1e-8770-3e4081e9c7d3
size(df) # (num_rows, num_columns)

# ╔═╡ 4c27de3c-bb06-4c98-b3c7-a69c69af529c
names(df) # names of columns

# ╔═╡ 187f2f8b-1e26-4b57-bec5-135b8dc7637b
histogram(df."Base Salary", title="Distribution of Base Salary", label="Employees Count")

# ╔═╡ ab1e4301-b614-4641-9f14-572dd6b0feb2
histogram(df[!,"Monthly Salary"])

# ╔═╡ aeae52f9-f4b3-451a-bc83-09d2d96b6d19
departments_df = combine(groupby(df, [:Department]), df -> 
        DataFrame(
            mean_monthly_salary = mean(df[!,"Monthly Salary"]),
            count_employees = nrow(df),
            std_monthly_salary = std(df[!,"Monthly Salary"])
        ))

# ╔═╡ 4532be5c-275d-4445-a3a4-b86e747b22c3
md"""
### Exploring Department
"""

# ╔═╡ a5258ead-99c4-4e43-92e1-b3cdf2d16a61
histogram(departments_count.count, title="Distribution of Department", xlabel="Employees Count", label="Department Count")

# ╔═╡ 095ea8de-8f51-4410-b48c-49df60176e84
md"""
<b>Observation</b>: There are 165 departments total. The top 5 largest departments consist of ~40% of the employees. The remaining 160 departments hold the other ~60%.
### Exploring Employee Status
<b>Observation</b>: The are 17 employement statuses total. 14 statuses have little employee counts (166) and 3 statuses have the lion's share (remaining 4170)
"""


# ╔═╡ 0eb51927-2903-499d-a911-826319149247


# ╔═╡ 3513525c-db20-4f5a-84ee-16569ea4ebd9


# ╔═╡ bc121ee0-30df-4542-b25f-7c6f51b8d6d2
md"""


### References

[^1]: Krishnamoorthy, Kalimuthu. Handbook of statistical distributions with applications. CRC Press, 2016. (p. 90)

"""

# ╔═╡ 0c5cac5f-a92a-426d-87bd-6c02d8e31626


# ╔═╡ 8ce5267a-b1cb-40b0-91ff-27851db5d3a8
df = transform(df, :"Base Salary" => ByRow(x -> x < 1000 ? 40 * 4 * x : x) => :"Monthly Salary");

# ╔═╡ b383b083-5099-475f-acc4-27c9501c542d
begin
	try # Running this block a second time throws an ArgumentError
	    rename!(df, "Department  " => "Department");
	catch
	    ArgumentError
	end
	
	df = df[:, ["Department", "Employee Status", "Base Salary"]]
	names(df)
end

# ╔═╡ 900c3192-c752-4bca-8f83-ba6c8eb56245
begin
	fpath = raw"C:\Users\noam\Repositories\bgu-abda.bitbucket.io\homework\02norfolk_employee_data.csv";
	df = CSV.read(fpath, DataFrame);
	first(df, 5)
end

# ╔═╡ Cell order:
# ╠═f4f05182-38b6-4bfc-bcbb-86ceb63cecbb
# ╟─cff83100-b955-11eb-2950-75483cd235df
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
# ╠═9fd59715-6aaf-4786-b417-39f795811e52
# ╠═70c61890-a65f-49a8-be91-9bb9ddbcbf00
# ╠═c53f412d-d763-4def-838b-517b5c4e9d81
# ╠═c89b51c7-e6ff-4cf7-9ab2-0b536784b686
# ╠═a63016f7-1954-4434-bc3f-bc8cb844057e
# ╠═64fd1f37-63f0-45ee-be3b-514b1193c7cb
# ╠═d5a0f661-41e6-4e02-b3ad-dd73ce1f30fc
# ╠═8cbd6987-c9a0-471e-85ad-e987e7383b12
# ╠═b31e0f83-2f64-44c6-998c-fd7be90cedd5
# ╠═fc4fccfa-00ae-431d-b5e7-ddb7e8cde96d
# ╟─7bc43c2b-4a41-4f3b-b193-875c7f558ce5
# ╠═5ff7c63f-b926-4c11-b219-dbc6948b9cd7
# ╠═78586ab6-e1c1-4731-aa2b-6bd73c1d6d64
# ╠═900c3192-c752-4bca-8f83-ba6c8eb56245
# ╠═b99d128a-6179-4644-8bff-8d7b23cc7a5e
# ╠═4c671e0d-6020-4a1e-8770-3e4081e9c7d3
# ╠═4c27de3c-bb06-4c98-b3c7-a69c69af529c
# ╠═197973a1-2cde-4be5-b4dc-608ec55f57f4
# ╠═b383b083-5099-475f-acc4-27c9501c542d
# ╠═00dae8e7-1358-4e49-a851-18812969199e
# ╠═187f2f8b-1e26-4b57-bec5-135b8dc7637b
# ╠═5c538050-fe6d-48bc-b206-e3bd45462cfe
# ╠═8ce5267a-b1cb-40b0-91ff-27851db5d3a8
# ╠═ab1e4301-b614-4641-9f14-572dd6b0feb2
# ╠═aeae52f9-f4b3-451a-bc83-09d2d96b6d19
# ╠═4532be5c-275d-4445-a3a4-b86e747b22c3
# ╠═a5258ead-99c4-4e43-92e1-b3cdf2d16a61
# ╠═095ea8de-8f51-4410-b48c-49df60176e84
# ╠═0eb51927-2903-499d-a911-826319149247
# ╠═3513525c-db20-4f5a-84ee-16569ea4ebd9
# ╟─bc121ee0-30df-4542-b25f-7c6f51b8d6d2
# ╠═0c5cac5f-a92a-426d-87bd-6c02d8e31626
