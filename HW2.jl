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
chn1_prior = sample(warp_breaks(warpbreaks_df.Breaks), Prior(), 10000)

# ╔═╡ d334e9d1-f55c-44ea-b3bc-5b7afb7df84c
plot(chn1_prior)

# ╔═╡ 58675fd9-d1fb-4d09-9879-12b495fa154a
describe(chn1_prior)

# ╔═╡ 520a5338-d39d-4a41-a133-f9257a6b312e
md"""
#### Sampling the Posterior
"""

# ╔═╡ 7b61939d-4fdd-4cf3-9396-bc669d79c69d
chn1 = sample(warp_breaks(warpbreaks_df.Breaks), NUTS(), MCMCThreads(), 10000, 4)

# ╔═╡ c1e5aa03-e300-477b-bd42-b4d0c14457b2
plot(chn1)

# ╔═╡ c7b066a2-cc0d-4d85-b7f9-a279b3345ee4
md"""

We can see that the sampler converges (all 4 chains are in agreement).

##### Seperating Wool Type
Let us estimate $λ_1$ seperately for each wool type:
"""

# ╔═╡ c8d43349-488d-4cd3-934b-88d334127272
begin
	chn1_A = sample(warp_breaks(warpbreaks_df[warpbreaks_df.Wool .== "A", :Breaks]), NUTS(), MCMCThreads(), 10000, 1)
	chn1_B = sample(warp_breaks(warpbreaks_df[warpbreaks_df.Wool .== "B", :Breaks]), NUTS(), MCMCThreads(), 10000, 1);
end

# ╔═╡ ff0dbf9b-0135-4c68-883b-ac30283ff9c7
begin
	plot(chn1_A, label="Wool A")
	plot!(chn1_B, label="Wool B", legend=:topright)
end

# ╔═╡ c5963b29-f843-4e38-ab21-51b3d891a197
md"""
## Plotting Prior and Posterior
Let's now plot the prior, posterior, and mean observation of $λ_1$:
"""

# ╔═╡ 88f73f6d-0709-46fb-b7a6-a9898a2f044c
begin
	density(chn1[:,:,1], lab="posterior", color=:red)  # A density plot of the 1st sampled chain
	vline!([mean(warpbreaks_df.Breaks)], linewidth = 2, color=:yellow, label="mean observation",)  # The mean observation
	density!(chn1_prior, label="prior",  legend=:topright, color=:cyan)  # The prior
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
	density(chn1_A, lab="posterior (wool A)", color=:pink)  # A density plot of the 1st sampled chain
	density!(chn1_B, lab="posterior (wool B)", color=:red)  # A density plot of the 1st sampled chain
	density!(chn1_prior, label="prior",  legend=:topright, color=:cyan)  # The prior
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
chn1_seperate = sample(warp_breaks_seperate(warpbreaks_df.Breaks, warpbreaks_df.Tension), NUTS(), MCMCThreads(), 10000, 4)

# ╔═╡ c89b51c7-e6ff-4cf7-9ab2-0b536784b686
plot(chn1_seperate)

# ╔═╡ a63016f7-1954-4434-bc3f-bc8cb844057e
inferred = mean(Array(group(chn1_seperate, :λ1)), dims = 1)'

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
			xlabel="Observed Breaks", ylabel="Inferred Mean Break Rate", title="Tension $t", subplot=i)
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

#### The Data

"""

# ╔═╡ 6e9e9d2a-db6f-4732-b8fb-293737f7a87d
begin
	fpath = raw"data/employee_salaries.csv"  # place your data here
	raw_norfolk_df = CSV.read(fpath, DataFrame)
	raw_norfolk_df = rename(raw_norfolk_df, "Department  " => "Department")  # minor cleanup
end

# ╔═╡ 197973a1-2cde-4be5-b4dc-608ec55f57f4
md"""
Thus, the dataset contains 4399 employees and 7 fields of information.
"""

# ╔═╡ d211f8d8-81f0-4950-a417-8f46b73cb079
md"""
### Data: Base Salary
The salary distribution is *clustered*! Employees have salaries in either the range (9,150) or in (3500, 260000).

We propose the assumption that the base salaries reflect either an hourly rate or a monthly income.

We assume that any value under 1000 is an hourly rate and any value at or above 1000 is a monthly salary. We will correct for this inconsistency in a new column "Monthly Salary". According to [^2], we assume a 40 hour work week. Notice the new minimal value is 1561, up from 9.5.
"""

# ╔═╡ f7f0254c-ea83-4ac9-a732-662b7d569008
histogram(raw_norfolk_df."Base Salary", xlabel="Base Salary", ylabel="Employees Count", label="raw data", title="Distribution of Base Salary")

# ╔═╡ 8a5b2359-8b2f-493a-8ec8-f9c32f251b8d
summarystats(raw_norfolk_df."Base Salary")

# ╔═╡ 900c3192-c752-4bca-8f83-ba6c8eb56245
norfolk_df = transform(raw_norfolk_df, :"Base Salary" => ByRow(x -> x < 1000 ? 40 * (52.15/12) * x : x) => :"Monthly Salary")  # add a monthly salary column

# ╔═╡ 16ea8e44-f818-4f7e-8db6-c2e7be14f067
histogram(norfolk_df."Monthly Salary", xlabel="Monthly Salary", ylabel="Employees Count", label="corrected data", title="Distribution of Monthly Salary")

# ╔═╡ 39257b51-2440-4dfe-93b8-7ae998f82135
summarystats(norfolk_df."Monthly Salary")

# ╔═╡ 4532be5c-275d-4445-a3a4-b86e747b22c3
md"""
### Data: Department
There are 165 departments total. The top 5 largest departments consist of ~40% of the employees. The remaining 160 departments hold the other ~60%.
The means and standard deviations are very different, 
"""

# ╔═╡ aeae52f9-f4b3-451a-bc83-09d2d96b6d19
begin
	departments_df = combine(groupby(norfolk_df, [:Department]), norfolk_df -> 
        DataFrame(
            mean_monthly_salary_dpt = mean(norfolk_df[!,"Monthly Salary"]),
            count_employees_dpt = nrow(norfolk_df),
            std_monthly_salary_dpt = std(norfolk_df[!,"Monthly Salary"])
        ))
	insertcols!(departments_df, 2,  :department_code =>1:nrow(departments_df))
end

# ╔═╡ 3be7aaac-d17d-48ea-ab59-57c90d938489
scatter(departments_df.std_monthly_salary_dpt, departments_df.count_employees_dpt, bins=20,xlabel="Salary Standard Deviation", ylabel="Employee Count", label="departments_df", title="All Departments")

# ╔═╡ 0a14d343-334f-4c00-bd8d-1dff602f04ab
scatter(departments_df.mean_monthly_salary_dpt, departments_df.count_employees_dpt, bins=20,xlabel="Mean Salary", ylabel="Employee Count", label="departments_df", title="All Departments")

# ╔═╡ feccab08-ff0a-4f27-9c3d-dace4ff7af02
histogram(departments_df.std_monthly_salary_dpt, bins=20,xlabel="Salary Standard Deviation", ylabel="Department Count", label="departments_df")

# ╔═╡ 4c4c2d10-f696-4eed-b430-6777b52f4b0d
histogram(departments_df.mean_monthly_salary_dpt, bins=20,xlabel="Mean Salary", ylabel="Department Count", label="departments_df")

# ╔═╡ 64651fec-5471-473c-9158-225e3a4d9585
md"""
The following dataframe holds extra information about the department.
"""

# ╔═╡ 095ea8de-8f51-4410-b48c-49df60176e84
md"""
### Data: Employee Status
The are 17 employement statuses total. 14 statuses have little employee counts (166) and 3 statuses have the lion's share (remaining 4170). Two departments have only one employee.
"""

# ╔═╡ a770c7e9-8b4c-4674-b303-b1f01b4bd287
status_df = combine(groupby(norfolk_df, [:"Employee Status"]), norfolk_df -> 
        DataFrame(
            mean_monthly_salary_status = mean(norfolk_df[!,"Monthly Salary"]),
            count_employees_status = nrow(norfolk_df),
            std_monthly_salary_status = std(norfolk_df[!,"Monthly Salary"])
        ))

# ╔═╡ 21476edb-54c7-4dd6-8ad3-876368c4c6c3
scatter(status_df.mean_monthly_salary_status, status_df.count_employees_status, bins=20,xlabel="Mean Salary", ylabel="Employee Count", label="status_df", title="All Employee Statuses")

# ╔═╡ 21d71472-7eef-48f7-89d7-c1b1a38ef047
scatter(status_df.std_monthly_salary_status, status_df.count_employees_status, bins=20,xlabel="Standard Deviation Salary", ylabel="Employee Count", label="status_df", title="All Employee Statuses")

# ╔═╡ a51cc963-8e5d-4984-8468-91e283d8585f
histogram(status_df.mean_monthly_salary_status, bins=20,xlabel="Mean Salary", ylabel="Status Count", label="status_df", title="All Employee Statuses")

# ╔═╡ 9c9fbfd9-e653-49a1-b45b-14620119b551
histogram(status_df.std_monthly_salary_status, bins=20,xlabel="Salary Standard Deviation", ylabel="Status Count", label="status_df", title="All Employee Statuses")

# ╔═╡ 95a6fbae-4aed-4660-9430-451d85dfbb5a
md"""
### Data: Extended Dataframe
The `extended_norfolk_df` DataFrame holds extra statistics regarding the department and status.
"""

# ╔═╡ 3ad2303c-f85a-41b4-ada3-dab8615fe558
begin
	inner_df = leftjoin(norfolk_df, departments_df, on=:Department)
	extended_norfolk_df = leftjoin(inner_df, status_df, on=:"Employee Status")
end

# ╔═╡ 7bc43c2b-4a41-4f3b-b193-875c7f558ce5
md"""
### Model 1
We ignore employee status and infer the salary distributions per department.



"""

# ╔═╡ 05aa5f83-eb21-4925-bb38-537f0664873b
@model function norfolk_departments(salaries, departments)
	n_departments = length(unique(departments)
	α ~ MvLogNormal(fill(0, n_departments), 1)
	μ ~ product_distribution(fill(Exponential(10), length(unique(departments))))
	for i in eachindex(salaries)
		salaries[i] ~ LogNormal(μ[departments[i]], σ[departments[i]])
	end
end

# ╔═╡ 57603099-e396-41be-9ce8-575a9f3dacce
chn2_prior = sample(norfolk_departments(norfolk_df."Monthly Salary", norfolk_df.Department), Prior(), 10000)

# ╔═╡ 51ab3fa3-fdc7-40f0-9dd2-d5d4be3f0f94
plot(chn2_prior)

# ╔═╡ bc121ee0-30df-4542-b25f-7c6f51b8d6d2
md"""


### References

[^1]: Krishnamoorthy, Kalimuthu. Handbook of statistical distributions with applications. CRC Press, 2016. (p. 90)

[^2]: Workweek and weekend. (2021, May 23). In Wikipedia. [https://en.wikipedia.org/wiki/Workweek\_and\_weekend](https://en.wikipedia.org/wiki/Workweek_and_weekend)

"""

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
# ╠═5ff7c63f-b926-4c11-b219-dbc6948b9cd7
# ╟─78586ab6-e1c1-4731-aa2b-6bd73c1d6d64
# ╠═6e9e9d2a-db6f-4732-b8fb-293737f7a87d
# ╟─197973a1-2cde-4be5-b4dc-608ec55f57f4
# ╟─d211f8d8-81f0-4950-a417-8f46b73cb079
# ╠═f7f0254c-ea83-4ac9-a732-662b7d569008
# ╠═8a5b2359-8b2f-493a-8ec8-f9c32f251b8d
# ╠═900c3192-c752-4bca-8f83-ba6c8eb56245
# ╠═16ea8e44-f818-4f7e-8db6-c2e7be14f067
# ╠═39257b51-2440-4dfe-93b8-7ae998f82135
# ╠═4532be5c-275d-4445-a3a4-b86e747b22c3
# ╠═aeae52f9-f4b3-451a-bc83-09d2d96b6d19
# ╠═3be7aaac-d17d-48ea-ab59-57c90d938489
# ╠═0a14d343-334f-4c00-bd8d-1dff602f04ab
# ╠═feccab08-ff0a-4f27-9c3d-dace4ff7af02
# ╠═4c4c2d10-f696-4eed-b430-6777b52f4b0d
# ╟─64651fec-5471-473c-9158-225e3a4d9585
# ╟─095ea8de-8f51-4410-b48c-49df60176e84
# ╠═a770c7e9-8b4c-4674-b303-b1f01b4bd287
# ╠═21476edb-54c7-4dd6-8ad3-876368c4c6c3
# ╠═21d71472-7eef-48f7-89d7-c1b1a38ef047
# ╠═a51cc963-8e5d-4984-8468-91e283d8585f
# ╠═9c9fbfd9-e653-49a1-b45b-14620119b551
# ╟─95a6fbae-4aed-4660-9430-451d85dfbb5a
# ╠═3ad2303c-f85a-41b4-ada3-dab8615fe558
# ╠═7bc43c2b-4a41-4f3b-b193-875c7f558ce5
# ╠═05aa5f83-eb21-4925-bb38-537f0664873b
# ╠═57603099-e396-41be-9ce8-575a9f3dacce
# ╠═51ab3fa3-fdc7-40f0-9dd2-d5d4be3f0f94
# ╟─bc121ee0-30df-4542-b25f-7c6f51b8d6d2
