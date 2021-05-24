### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ f4f05182-38b6-4bfc-bcbb-86ceb63cecbb
using Distributions, Turing, StatsPlots, Random, RDatasets, CSV, DataFrames, CategoricalArrays, Dates

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
	transform!(warpbreaks_df, [:Wool, :Tension] => ((w, t) -> categorical(string.(w,t))) => "wool_tension")
end

# ╔═╡ e597b2c1-82d9-44fc-bcb8-20de1a15269d
categorical(["A", "B"])

# ╔═╡ 8b8e093c-1a42-40b1-8775-53cdf9cde072
warpbreaks_df

# ╔═╡ 8e981ff6-e0a1-45f6-98ef-113432c13808
histogram(warpbreaks_df.Breaks, bins=20,xlabel="Count of Warp Breaks", ylabel="Count of Looms", lab="all weaves", title="Weaves")

# ╔═╡ b44ae535-e44e-4dbc-a97e-bb50ca90e2bb
summarystats(warpbreaks_df.Breaks), std(warpbreaks_df.Breaks)

# ╔═╡ fa5603e5-7d9d-453a-9c96-7d6a77f12127
md"""

#### Data: Wool

"""

# ╔═╡ 8912c50a-3b43-4f0f-91ad-5c2a10f9d1a1
let
	bins = 0:5:100
	xlim = (0,100)
	@df warpbreaks_df groupedhist(:Breaks, group=:Wool, bar_position = :dodge, bins=bins, xlabel="Warp Breaks", ylabel="Weave Counts", title="Warp Breaks by Wool")
end

# ╔═╡ 60951dfa-e531-4fcd-ac2f-eab1bd7ba90b
summarystats(warpbreaks_df.Breaks), summarystats(warpbreaks_df[warpbreaks_df.Wool .== "A",:].Breaks), summarystats(warpbreaks_df[warpbreaks_df.Wool .== "B",:].Breaks)

# ╔═╡ c5d27891-067c-46b0-90b7-bcdae15e6b8c
md"""
Observate that on average, **wool of type A breaks more often than wool of type B**.

"""

# ╔═╡ 05541d77-be89-4a0d-ac7e-71738c7d12b1
md"""

#### Data: Tension

"""

# ╔═╡ 3b5c9837-4a12-4bc5-81b1-7b425a640465
let
	bins = 0:5:100
	xlim = (0,100)
	@df warpbreaks_df groupedhist(:Breaks, group=:Tension, bar_position = :dodge, bins=bins, xlabel="Warp Breaks", ylabel="Weave Counts", title="Warp Breaks by Tension")
end

# ╔═╡ 5ac5e359-6964-41e4-a7ee-fb85756f130a
let
	bins = 0:5:100
	xlim = (0,100)
	@df warpbreaks_df groupedhist(:Breaks, group=:wool_tension, bar_position = :dodge, bins=bins, xlabel="Warp Breaks", ylabel="Weave Counts", title="Warp Breaks by Tension")
end

# ╔═╡ 24ebf4be-0b4e-4158-ba05-b71ddfec3c44
md"""
## Model 1: Fully Pooled

The Poisson distribution gives the probability of observing some $k \in \mathbb{N}$ events in a given period of time, assuming that events occur independently at a constant rate. According to [^1], it can be used to model the distribution of the number of defects in a piece of material. Since every break in the wool is caused independently with no memory of the previous breaks, we will use the Poisson distribution to model the number of warp breaks in a loom:

$λ_0 = 27$
$λ_A ∼ Exponential(λ_0)$
$λ_B ∼ Exponential(λ_0)$
$obs[i] ∼ Poisson(λ_{W_i}) \space \forall i$

Where $obs[i]$ is the number of breaks in the $i$'th loom, and $W_i$ is its wool type.
"""

# ╔═╡ 9ff9b9b0-3438-438d-920f-efb32a27cbca
@model function warp_breaks(breaks, wool)
	λ0 = 27
	λA ~ Exponential(λ0)
	λB ~ Exponential(λ0)
	for i in eachindex(breaks)
		if wool[i] == "A"
			breaks[i] ~ Poisson(λA)
		elseif wool[i] == "B"
			breaks[i] ~ Poisson(λB)
		else
			throw(ErrorException("Invalid Wool Type"))
		end
	end
end

# ╔═╡ 6c478f37-ecb9-4a54-a252-8bfc21251a24
md"""
#### Sampling the Prior
"""

# ╔═╡ 4d6cf4f7-961a-4fb9-8ea6-5babe84cafa7
chn1_pooled_prior = sample(warp_breaks(warpbreaks_df.Breaks, warpbreaks_df.Wool), Prior(), 10000)

# ╔═╡ 58675fd9-d1fb-4d09-9879-12b495fa154a
describe(chn1_pooled_prior)

# ╔═╡ d334e9d1-f55c-44ea-b3bc-5b7afb7df84c
plot(chn1_pooled_prior)

# ╔═╡ 520a5338-d39d-4a41-a133-f9257a6b312e
md"""
#### Sampling the Posterior
"""

# ╔═╡ 7b61939d-4fdd-4cf3-9396-bc669d79c69d
chn1_pooled = sample(warp_breaks(warpbreaks_df.Breaks, warpbreaks_df.Wool), NUTS(), MCMCThreads(), 10000, 4)

# ╔═╡ 5da2340b-0ef1-4454-83ba-8557fada0b98
describe(chn1_pooled)

# ╔═╡ 58123c1a-8ae1-4858-b847-b842a07fd2f8
mean(chn1_pooled[:λB] ./ chn1_pooled[:λA])

# ╔═╡ 702a8463-f89f-4fd7-8a7a-3053bebd5f28
std(chn1_pooled[:λB] ./ chn1_pooled[:λA])

# ╔═╡ c1e5aa03-e300-477b-bd42-b4d0c14457b2
plot(chn1_pooled)

# ╔═╡ c7b066a2-cc0d-4d85-b7f9-a279b3345ee4
md"""

We can see that the sampler converges (all 4 chains are in agreement). 
"""

# ╔═╡ c5963b29-f843-4e38-ab21-51b3d891a197
md"""
#### Plotting Prior and Posterior
The posterior is in agreement with the data: mean warp breaks with wool type B is less than mean warp breaks with wool of type A, and there is a strong mode at the observed mean.
Let's now plot the prior, posterior, and mean observation of $λ_A, λ_B$:
"""

# ╔═╡ 88f73f6d-0709-46fb-b7a6-a9898a2f044c
begin
	density(chn1_pooled[:,:,1], lab="posterior", color=:red)  # A density plot of the 1st sampled chain
	density!(chn1_pooled_prior, label="prior",  legend=:topright, color=:cyan)  # The prior
	vline!([mean(warpbreaks_df[warpbreaks_df.Wool .== "A",:Breaks])], linewidth = 2, subplot=1, color=:yellow, label="mean observation for wool A",)  # The mean observation for wool A
	vline!([mean(warpbreaks_df[warpbreaks_df.Wool .== "B",:Breaks])], linewidth = 2, subplot=2, color=:yellow, label="mean observation for wool B",)  # The mean observation for wool B
end

# ╔═╡ b2f5b368-2940-442d-936f-58aec32c889e
md"""
The fact that the prior is so flat compared to the posterior alludes to the fact that it is a rather uninformative prior.
"""

# ╔═╡ 9fd59715-6aaf-4786-b417-39f795811e52
md"""
## Model 2: Fully Seperate
Here, we account for warp tension $T$ as well. We let each parameter $W, T$ have it's own set of parameters (not shared).

TODO: Fix this description

$λ_0 = 27$
$λ_1[W] ∼ Exponential(λ_0) \space \forall W \in \{A, B\}$
$λ_2[T] ∼ Exponential(λ_0) \space \forall T \in \{L, M, H\}$
$obs[i] ∼ Poisson(λ_1[W_i]) + Poisson(λ_2[T_i]) \space \forall i$

Where $obs[i]$ is the number of breaks in the $i$'th loom.

"""

# ╔═╡ 70c61890-a65f-49a8-be91-9bb9ddbcbf00
@model function warp_breaks_seperate(breaks, wool, tension)
	λ0 = 27
	λAL ~ Exponential(λ0)
	λAM ~ Exponential(λ0)
	λAH ~ Exponential(λ0)
	λBL ~ Exponential(λ0)
	λBM ~ Exponential(λ0)
	λBH ~ Exponential(λ0)
	
	for i in eachindex(breaks)
		if (wool[i] == "A") .& (tension[i] == "L")
			λ1 = λAL
		elseif (wool[i] == "A") .& (tension[i] == "M")
			λ1 = λAM
		elseif (wool[i] == "A") .& (tension[i] == "H")
			λ1 = λAH
		elseif (wool[i] == "B") .& (tension[i] == "L")
			λ1 = λBL
		elseif (wool[i] == "B") .& (tension[i] == "M")
			λ1 = λBM
		elseif (wool[i] == "B") .& (tension[i] == "H")
			λ1 = λBH
		end
		breaks[i] ~ Poisson(λ1)
	end
end

# ╔═╡ 6c582d07-1dcd-4896-bc92-cd419c8ef78c
md"""
#### Sampling the Prior
"""

# ╔═╡ fa906b36-9d5a-45f9-ba21-93276e663e62
chn1_seperate_prior = sample(warp_breaks_seperate(warpbreaks_df.Breaks, warpbreaks_df.Wool, warpbreaks_df.Tension), Prior(), 10000)

# ╔═╡ 57e2e8c3-653d-46c2-88a1-bb3fcae985eb
describe(chn1_seperate_prior)

# ╔═╡ f247548c-d548-46b9-bc48-c39c7818fbd3
plot(chn1_seperate_prior)

# ╔═╡ 0c0328b0-57db-4958-a80a-9911d40a1382
md"""
#### Sampling the Posterior
"""

# ╔═╡ c53f412d-d763-4def-838b-517b5c4e9d81
chn1_seperate = sample(warp_breaks_seperate(warpbreaks_df.Breaks, warpbreaks_df.Wool, warpbreaks_df.Tension), NUTS(), MCMCThreads(), 16000, 4)

# ╔═╡ 73bf09d1-f8bd-49a5-aa2e-fc69b6ccaa44
describe(chn1_seperate)

# ╔═╡ c89b51c7-e6ff-4cf7-9ab2-0b536784b686
plot(chn1_seperate)

# ╔═╡ 7b11a82c-dafd-450e-9db2-b6c6a50ba114
md"""
#### Plotting Prior and Posterior
We plot the prior and posterior distributions, and overlay the mean/2 break rate. We  divide by 2 because in this new model, the process is modelled as a sum of two processes.

We can see that compared to the first model, the posteriors of $λ_A, λ_B$ have higher variance (i.e. less confident).

This could be because there are more parameters, so we have less certainty per each parameter.
"""

# ╔═╡ e4d63042-80d5-4bf0-82e9-a961f6b56d29
let	
# 	todo: Fix VLines
	gdf = groupby(warpbreaks_df, [:Wool, :Tension])
	means = combine(gdf, :Breaks => mean)
	
	density(chn1_seperate[:,:,1], lab="posterior", color=:red)  # A density plot of the 1st sampled chain
	density!(chn1_seperate_prior, label="prior",  legend=:topright, color=:cyan)  # The prior
	vline!([mean(warpbreaks_df[(warpbreaks_df.Wool .== "A") .& (warpbreaks_df.Tension .== "H"), :Breaks])], linewidth = 2, subplot=1, color=:yellow, label="mean observation for wool A, tension H",)  # The mean observation for wool A
	vline!([mean(warpbreaks_df[warpbreaks_df.Wool .== "B", :Breaks])/2], linewidth = 2, subplot=2, color=:yellow, label="mean/2 observation for wool B",)  # The mean observation for wool B
	vline!([mean(warpbreaks_df[warpbreaks_df.Tension .== "H", :Breaks])/2], linewidth = 2, subplot=3, color=:yellow, label="mean/2 observation for tension H",)  # The mean observation for tension H
	vline!([mean(warpbreaks_df[warpbreaks_df.Tension .== "L", :Breaks])/2], linewidth = 2, subplot=4, color=:yellow, label="mean/2 observation for tension L",)  # The mean observation for tension L
	vline!([mean(warpbreaks_df[warpbreaks_df.Tension .== "M", :Breaks])/2], linewidth = 2, subplot=5, color=:yellow, label="mean/2 observation for tension M",)  # The mean observation for tension M
end

# ╔═╡ fc4fccfa-00ae-431d-b5e7-ddb7e8cde96d
md"""
## Model 3: Hierarchical
We would like to account for wool tension $T \in \{L, M, H\}$ in a multilevel model.

$λ_0 = 27$

$μ = 5$

$σ = 1$


$λ_1 ∼ Exponential(λ_0)$

$obs[i] ∼ Poisson(λ_1) \space \forall i$

Where $obs[i]$ is the number of breaks in the $i$'th loom.

"""

# ╔═╡ 5ff7c63f-b926-4c11-b219-dbc6948b9cd7
@model function warp_breaks_hier(breaks, wool, tension)
	λ0 = 27
	λAL ~ Exponential(λ0)
	λAM ~ Exponential(λ0)
	λAH ~ Exponential(λ0)
	
	Δ ~ Uniform(0, 10)
	
	for i in eachindex(breaks)
		if (wool[i] == "A") .& (tension[i] == "L")
			λ1 = λAL
		elseif (wool[i] == "A") .& (tension[i] == "M")
			λ1 = λAM
		elseif (wool[i] == "A") .& (tension[i] == "H")
			λ1 = λAH
		elseif (wool[i] == "B") .& (tension[i] == "L")
			λ1 = λAL * Δ
		elseif (wool[i] == "B") .& (tension[i] == "M")
			λ1 = λAM * Δ
		elseif (wool[i] == "B") .& (tension[i] == "H")
			λ1 = λAH * Δ
		end
		breaks[i] ~ Poisson(λ1)
	end
end

# ╔═╡ 02638f99-7697-4357-9fd4-25759f69b135
md"""
#### Sampling the Prior
"""

# ╔═╡ 0f737394-034e-442a-9f85-08e1987caf0f
chn1_hier_prior = sample(warp_breaks_hier(warpbreaks_df.Breaks, warpbreaks_df.Wool, warpbreaks_df.Tension), Prior(), 10000)

# ╔═╡ 81db3224-542b-414b-b5db-3a630f3ef4fb
describe(chn1_hier_prior)

# ╔═╡ ca435de3-91f6-498b-8710-103fb4402385
plot(chn1_hier_prior)

# ╔═╡ 3dc511b4-a77c-4224-9726-375cc49e2aea
md"""
#### Sampling the Posterior
"""

# ╔═╡ ed2ed1b9-8b18-41d2-b998-3d73d4073018
chn1_hier = sample(warp_breaks_hier(warpbreaks_df.Breaks, warpbreaks_df.Wool, warpbreaks_df.Tension), NUTS(), MCMCThreads(), 10000, 4)

# ╔═╡ a939664d-e678-4f96-839c-957a356f55c7
describe(chn1_hier)

# ╔═╡ d4854857-3c15-47c2-aa26-8169889cb101
plot(chn1_hier)

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
histogram(raw_norfolk_df."Base Salary", xlabel="Base Salary", ylabel="Employees Count", label="raw data", title="All Employees")

# ╔═╡ 8a5b2359-8b2f-493a-8ec8-f9c32f251b8d
summarystats(raw_norfolk_df."Base Salary")

# ╔═╡ 900c3192-c752-4bca-8f83-ba6c8eb56245
begin
	norfolk_df = transform(raw_norfolk_df, :"Base Salary" => ByRow(x -> x < 1000 ? 40 * 52.15 * x : x) => :"ysalary")  # add a yearly salary column
	transform!(norfolk_df, :"Initial Hire Date" => (d -> Date.(d, "m/d/y")) => :"initial_hire_date")
	transform!(norfolk_df, :"Date in Position" => (d -> Date.(d, "m/d/y")) => :"date_in_position")
		transform!(norfolk_df, :"Fair Labor Standards Act (FLSA) " => (f -> categorical(f)) => :"flsa")
end

# ╔═╡ a97446f0-5423-489d-9a81-ba9d54fd2a93
let
	s = norfolk_df."Initial Hire Date"[1]
	Date(s,"m/d/y")
end

# ╔═╡ 16ea8e44-f818-4f7e-8db6-c2e7be14f067
histogram(norfolk_df.ysalary, xlabel="Salary", ylabel="Employees Count", label="corrected data", title="All Employees")

# ╔═╡ 39257b51-2440-4dfe-93b8-7ae998f82135
summarystats(norfolk_df.ysalary)

# ╔═╡ 807bc580-ab4d-4f98-b40c-1ab3cb62faeb
md"""
#### Data: Initial Hire Date
"""

# ╔═╡ 5a65d06f-36c3-4c7a-a9da-5cfa0c2a3c33
scatter(norfolk_df.ysalary, norfolk_df.initial_hire_date, markersize=1)

# ╔═╡ 84493a57-6082-4a28-bf17-22053e322b83
md"""
#### Data: Date in Position
It is visible that the more senior employees have a higher average salary.
"""


# ╔═╡ 4e57b52e-79cd-4102-98af-5675abfd6442
scatter(norfolk_df.ysalary, norfolk_df.date_in_position, markersize=1)

# ╔═╡ 32be7e4a-4827-4cee-989b-76169c91a3d5
md"""
#### Data: Fair Labor Standards Act (FLSA)
Employees who are exempt from FLSA have a higher average salary.
"""

# ╔═╡ 65cfb4dc-7e7a-49e9-95bf-fd99cdead502
let
	@df norfolk_df groupedhist(:ysalary, group=:flsa, bar_position = :dodge, xlabel="Yearly Salary", ylabel="Employee Count", title="Employees by FLSA")
end

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
            mean_ysalary_dpt = mean(norfolk_df[!,"ysalary"]),
            count_employees_dpt = nrow(norfolk_df),
            std_ysalary_dpt = std(norfolk_df[!,"ysalary"])
        ))
	insertcols!(departments_df, 2,  :department_code =>1:nrow(departments_df))
end

# ╔═╡ 17301e96-274c-4cb6-bda3-ed7623fc04bb
sort(departments_df, :mean_ysalary_dpt)

# ╔═╡ 0a14d343-334f-4c00-bd8d-1dff602f04ab
scatter(departments_df.mean_ysalary_dpt, departments_df.count_employees_dpt, bins=20,xlabel="Mean Salary", ylabel="Employee Count", label="departments_df", title="All Departments")

# ╔═╡ feccab08-ff0a-4f27-9c3d-dace4ff7af02
histogram(departments_df.std_ysalary_dpt, bins=20,xlabel="Salary Standard Deviation", ylabel="Department Count", label="departments_df")

# ╔═╡ 4c4c2d10-f696-4eed-b430-6777b52f4b0d
histogram(departments_df.mean_ysalary_dpt, bins=20, xlabel="Mean Salary", ylabel="Department Count", label="departments_df")

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
            mean_monthly_salary_status = mean(norfolk_df[!,:ysalary]),
            count_employees_status = nrow(norfolk_df),
            std_monthly_salary_status = std(norfolk_df[!,:ysalary])
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
# ╠═e597b2c1-82d9-44fc-bcb8-20de1a15269d
# ╠═8b8e093c-1a42-40b1-8775-53cdf9cde072
# ╠═8e981ff6-e0a1-45f6-98ef-113432c13808
# ╠═b44ae535-e44e-4dbc-a97e-bb50ca90e2bb
# ╟─fa5603e5-7d9d-453a-9c96-7d6a77f12127
# ╠═8912c50a-3b43-4f0f-91ad-5c2a10f9d1a1
# ╠═60951dfa-e531-4fcd-ac2f-eab1bd7ba90b
# ╟─c5d27891-067c-46b0-90b7-bcdae15e6b8c
# ╟─05541d77-be89-4a0d-ac7e-71738c7d12b1
# ╠═3b5c9837-4a12-4bc5-81b1-7b425a640465
# ╠═5ac5e359-6964-41e4-a7ee-fb85756f130a
# ╟─24ebf4be-0b4e-4158-ba05-b71ddfec3c44
# ╠═9ff9b9b0-3438-438d-920f-efb32a27cbca
# ╟─6c478f37-ecb9-4a54-a252-8bfc21251a24
# ╠═4d6cf4f7-961a-4fb9-8ea6-5babe84cafa7
# ╠═58675fd9-d1fb-4d09-9879-12b495fa154a
# ╠═d334e9d1-f55c-44ea-b3bc-5b7afb7df84c
# ╟─520a5338-d39d-4a41-a133-f9257a6b312e
# ╠═7b61939d-4fdd-4cf3-9396-bc669d79c69d
# ╠═5da2340b-0ef1-4454-83ba-8557fada0b98
# ╠═58123c1a-8ae1-4858-b847-b842a07fd2f8
# ╠═702a8463-f89f-4fd7-8a7a-3053bebd5f28
# ╠═c1e5aa03-e300-477b-bd42-b4d0c14457b2
# ╟─c7b066a2-cc0d-4d85-b7f9-a279b3345ee4
# ╟─c5963b29-f843-4e38-ab21-51b3d891a197
# ╠═88f73f6d-0709-46fb-b7a6-a9898a2f044c
# ╟─b2f5b368-2940-442d-936f-58aec32c889e
# ╠═9fd59715-6aaf-4786-b417-39f795811e52
# ╠═70c61890-a65f-49a8-be91-9bb9ddbcbf00
# ╟─6c582d07-1dcd-4896-bc92-cd419c8ef78c
# ╠═fa906b36-9d5a-45f9-ba21-93276e663e62
# ╠═57e2e8c3-653d-46c2-88a1-bb3fcae985eb
# ╠═f247548c-d548-46b9-bc48-c39c7818fbd3
# ╟─0c0328b0-57db-4958-a80a-9911d40a1382
# ╠═c53f412d-d763-4def-838b-517b5c4e9d81
# ╠═73bf09d1-f8bd-49a5-aa2e-fc69b6ccaa44
# ╠═c89b51c7-e6ff-4cf7-9ab2-0b536784b686
# ╟─7b11a82c-dafd-450e-9db2-b6c6a50ba114
# ╠═e4d63042-80d5-4bf0-82e9-a961f6b56d29
# ╠═fc4fccfa-00ae-431d-b5e7-ddb7e8cde96d
# ╠═5ff7c63f-b926-4c11-b219-dbc6948b9cd7
# ╟─02638f99-7697-4357-9fd4-25759f69b135
# ╠═0f737394-034e-442a-9f85-08e1987caf0f
# ╠═81db3224-542b-414b-b5db-3a630f3ef4fb
# ╠═ca435de3-91f6-498b-8710-103fb4402385
# ╠═3dc511b4-a77c-4224-9726-375cc49e2aea
# ╠═ed2ed1b9-8b18-41d2-b998-3d73d4073018
# ╠═a939664d-e678-4f96-839c-957a356f55c7
# ╠═d4854857-3c15-47c2-aa26-8169889cb101
# ╟─78586ab6-e1c1-4731-aa2b-6bd73c1d6d64
# ╠═6e9e9d2a-db6f-4732-b8fb-293737f7a87d
# ╟─197973a1-2cde-4be5-b4dc-608ec55f57f4
# ╟─d211f8d8-81f0-4950-a417-8f46b73cb079
# ╠═f7f0254c-ea83-4ac9-a732-662b7d569008
# ╠═8a5b2359-8b2f-493a-8ec8-f9c32f251b8d
# ╠═900c3192-c752-4bca-8f83-ba6c8eb56245
# ╠═a97446f0-5423-489d-9a81-ba9d54fd2a93
# ╠═16ea8e44-f818-4f7e-8db6-c2e7be14f067
# ╠═39257b51-2440-4dfe-93b8-7ae998f82135
# ╟─807bc580-ab4d-4f98-b40c-1ab3cb62faeb
# ╠═5a65d06f-36c3-4c7a-a9da-5cfa0c2a3c33
# ╟─84493a57-6082-4a28-bf17-22053e322b83
# ╠═4e57b52e-79cd-4102-98af-5675abfd6442
# ╟─32be7e4a-4827-4cee-989b-76169c91a3d5
# ╠═65cfb4dc-7e7a-49e9-95bf-fd99cdead502
# ╟─4532be5c-275d-4445-a3a4-b86e747b22c3
# ╠═aeae52f9-f4b3-451a-bc83-09d2d96b6d19
# ╠═17301e96-274c-4cb6-bda3-ed7623fc04bb
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
