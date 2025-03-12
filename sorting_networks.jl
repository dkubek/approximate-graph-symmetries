### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 4b346729-aafc-40c7-af93-2073ee7e2dcf
begin
	using JuMP
	using Optimization
	using OptimizationMOI
	using Ipopt

	using SparseArrays
	using LinearAlgebra

	using Random
end

# ╔═╡ faf40b51-c850-430c-bf59-eb99da68631a
begin
	using NPZ
	
	# Define the Instance struct with node count
	struct Instance
	    graph_type::String
	    path::String
	    n::Int
	end
	
	function collect_instances(instance_folder::String)
	    instances = Instance[]
	    
	    # Check if folder exists
	    if !isdir(instance_folder)
	        error("Instance folder '$instance_folder' does not exist.")
	    end
	    
	    # Iterate through subdirectories
	    for graph_type_folder in readdir(instance_folder, join=true)
	        if isdir(graph_type_folder)
	            graph_type = basename(graph_type_folder)
	            
	            # Iterate through files in the subdirectory
	            for instance_path in readdir(graph_type_folder, join=true)
	                if isfile(instance_path) && endswith(instance_path, ".npz") && !contains(instance_path, "_allInfo")
	                    # Extract n_nodes from filename
	                    filename = basename(instance_path)
	                    n_nodes_match = match(r"nNodes(\d+)", filename)
	                    n_nodes = n_nodes_match !== nothing ? parse(Int, n_nodes_match.captures[1]) : 0
	                    
	                    push!(instances, Instance(graph_type, instance_path, n_nodes))
	                end
	            end
	        end
	    end
	    
	    return instances
	end
	
	"""
	    load_instance_data(instance::Instance) -> Dict
	
	Loads data from a .npz file and returns it as a dictionary.
	"""
	function load_instance_data(instance::Instance)
	    if !isfile(instance.path)
	        error("Instance file '$(instance.path)' does not exist.")
	    end
	    
	    try
	        data = npzread(instance.path)
	        return data
	    catch e
	        error("Failed to load instance data from '$(instance.path)': $e")
	    end
	end
end

# ╔═╡ a69c0df2-fce7-11ef-14c8-ad51bfb0e79d
md"""
# Comparator Network
"""

# ╔═╡ fb33e25d-f5c6-4faf-8884-d853aca3b1a1
"""
    Comparator

Represents a comparator in a sorting network.
`i` and `j` are the indices of the two wires being compared.
"""
struct Comparator
    i::Int
    j::Int
end

# ╔═╡ e8589476-2744-42d2-8ef5-36f705c84a1e
"""
    bitonicMerge(lo, hi, dir)

Merge a bitonic sequence from index `lo` to `hi`.
If `dir` is true, sort in ascending order; otherwise, sort in descending order.
"""
function bitonicMerge(lo, hi, dir)
    if hi - lo > 1
        mid = lo + div(hi - lo, 2)
        comparators = Comparator[]
        
        # Create comparators between corresponding elements in the two halves
        for i in lo:(mid-1)
            if dir
                append!(comparators, [Comparator(i, i + (mid - lo))])
            else
                append!(comparators, [Comparator(i + (mid - lo), i)])
            end
        end
        
        # Recursively merge both halves
        append!(comparators, bitonicMerge(lo, mid, dir))
        append!(comparators, bitonicMerge(mid, hi, dir))
        
        return comparators
    end
    return Comparator[]
end


# ╔═╡ 894e8680-a849-45dd-8edb-91136986b256
"""
    bitonicSort(lo, hi, dir)

Generate comparators for a bitonic sort on elements from index `lo` to `hi`.
If `dir` is true, sort in ascending order; otherwise, sort in descending order.
"""
function bitonicSort(lo, hi, dir)
    if hi - lo > 1
        mid = lo + div(hi - lo, 2)
        # Recursively sort both halves
        comparators1 = bitonicSort(lo, mid, !dir)
        comparators2 = bitonicSort(mid, hi, dir)
        # Merge the two halves
        comparators3 = bitonicMerge(lo, hi, dir)
        return vcat(comparators1, comparators2, comparators3)
    end
    return Comparator[]
end

# ╔═╡ 74d2e80d-f9f4-4bea-8643-93f5934eed01
"""
    generateBatcherNetwork(n)

Generate a complete Batcher's bitonic sorting network for n inputs.
Returns a list of comparators.
"""
function generateBatcherNetwork(n)
    # Round up to the nearest power of 2
    m = 2^ceil(Int, log2(n))
    
    # Generate the network for the power of 2
    comparators = bitonicSort(0, m, true)
    
    # Filter out comparators that involve inputs beyond n
    return filter(c -> c.i < n && c.j < n, comparators)
end

# ╔═╡ 82c3000e-4ae2-44a5-95e7-95e8c90fa4cc
"""
    applySortingNetwork(input, comparators)

Apply a sorting network defined by comparators to the input sequence.
Returns the sorted output.
"""
function applySortingNetwork(input, comparators)
    # Make a copy of the input to avoid modifying the original
    values = copy(input)
    
    # Apply each comparator in order
    for comp in comparators
        i, j = comp.i + 1, comp.j + 1  # Adjust for 1-based indexing in Julia
        
        # If the values are in the wrong order, swap them
        if values[i] > values[j]
            values[i], values[j] = values[j], values[i]
        end
    end
    
    return values
end


# ╔═╡ 97ba189f-5d3d-406d-bc19-06059810a950
network = generateBatcherNetwork(10)

# ╔═╡ 03c283a3-e26e-4e64-b0ca-c315cf2fa8e8
function generateInitialPoint(n, comparators)
    num_comparators = length(comparators)
    num_vars = n + 2 * num_comparators
    
    x0 = zeros(num_vars)
    
    # Set input values to random permutation of 1:n
    input_values = shuffle(1:n)
    x0[1:n] = input_values
    
    # Track wire values
    wires = copy(input_values)
    
    # Process each comparator to determine intermediate values
    for (k, comp) in enumerate(comparators)
        i, j = comp.i + 1, comp.j + 1
        
        # Get wire values
        top_in = wires[i]
        bottom_in = wires[j]
        
        # Calculate sorted outputs
        top_out = min(top_in, bottom_in)
        bottom_out = max(top_in, bottom_in)
        
        # Set the variable values
        x0[n + 2*k - 1] = top_out
        x0[n + 2*k] = bottom_out
        
        # Update wires
        wires[i] = top_out
        wires[j] = bottom_out
    end
    
    return x0
end

# ╔═╡ af3bffaa-3e93-4a76-8f35-31122920ca53
"""
    sortingNetworkPolytope(n)
Create a sorting network polytope for n inputs in a form compatible with Optimization.jl.
Returns variables, constraints as sparse matrices, and a mapping for reference.
"""
function sortingNetworkPolytope(n)
    
    # Generate the sorting network
    comparators = generateBatcherNetwork(n)
    
    # Create variables
    num_comparators = length(comparators)
    num_vars = n + 2 * num_comparators  # input vars + 2 output vars per comparator
    
    # Dictionary for wire values
    wires = zeros(Int, n)
    
    # Initialize input variables with indices
    for i in 1:n
        wires[i] = i
    end
    
    # For building sparse matrices
    eq_rows = Int64[]
    eq_cols = Int64[]
    eq_vals = Float64[]
    eq_b = Float64[]
    
    ineq_rows = Int64[]
    ineq_cols = Int64[]
    ineq_vals = Float64[]
    ineq_b = Float64[]
    
    eq_row_counter = 1
    ineq_row_counter = 1
    
    # Process each comparator
    for (k, comp) in enumerate(comparators)
        i, j = comp.i + 1, comp.j + 1
        
        # Indices for the output variables of this comparator
        top_out_idx = n + 2*k - 1
        bottom_out_idx = n + 2*k
        
		# Get the current input wire indices
        top_in_idx = wires[i]
        bottom_in_idx = wires[j]
        
        # Constraint 1: top_in + bottom_in - top_out - bottom_out = 0
        push!(eq_rows, eq_row_counter)
        push!(eq_cols, top_in_idx)
        push!(eq_vals, 1.0)
        
        push!(eq_rows, eq_row_counter)
        push!(eq_cols, bottom_in_idx)
        push!(eq_vals, 1.0)
        
        push!(eq_rows, eq_row_counter)
        push!(eq_cols, top_out_idx)
        push!(eq_vals, -1.0)
        
        push!(eq_rows, eq_row_counter)
        push!(eq_cols, bottom_out_idx)
        push!(eq_vals, -1.0)
        
        push!(eq_b, 0.0)
        eq_row_counter += 1
        
        # Constraint 2: top_out - top_in <= 0
        push!(ineq_rows, ineq_row_counter)
        push!(ineq_cols, top_out_idx)
        push!(ineq_vals, 1.0)
        
        push!(ineq_rows, ineq_row_counter)
        push!(ineq_cols, top_in_idx)
        push!(ineq_vals, -1.0)
        
        push!(ineq_b, 0.0)
        ineq_row_counter += 1
        
        # Constraint 3: top_out - bottom_in <= 0
        push!(ineq_rows, ineq_row_counter)
        push!(ineq_cols, top_out_idx)
        push!(ineq_vals, 1.0)
        
        push!(ineq_rows, ineq_row_counter)
        push!(ineq_cols, bottom_in_idx)
        push!(ineq_vals, -1.0)
        
        push!(ineq_b, 0.0)
        ineq_row_counter += 1
        
        # Update wire values
        wires[i] = top_out_idx
        wires[j] = bottom_out_idx
    end
    
    # Output constraints: final values should be 1,2,...,n
    for i in 1:n
        final_var_idx = wires[i]
        
        push!(eq_rows, eq_row_counter)
        push!(eq_cols, final_var_idx)
        push!(eq_vals, 1.0)
        
        push!(eq_b, Float64(i))
        eq_row_counter += 1
    end
    
    # Create sparse matrices
    num_eq_constraints = eq_row_counter - 1
    num_ineq_constraints = ineq_row_counter - 1
    
    equality_A = sparse(eq_rows, eq_cols, eq_vals, num_eq_constraints, num_vars)
    equality_b = Vector{Float64}(eq_b)
    
    inequality_A = sparse(ineq_rows, ineq_cols, ineq_vals, num_ineq_constraints, num_vars)
    inequality_b = Vector{Float64}(ineq_b)
    
    return num_vars, (equality_A=equality_A, equality_b=equality_b, 
                     inequality_A=inequality_A, inequality_b=inequality_b), comparators
end

# ╔═╡ 747bfc6e-4d9d-48cd-9d13-ae272ae5d460
begin
	"""
	    periodic(f::Function, period::Integer)
	
	Create a periodic version of function f with given period.
	"""
	function periodic(f::Function, period::Integer)
		# Enough for our use case to repeat 3 times
	    return x -> f(x + period) + f(x) + f(x - period)
	end
	
	
	"""
	    irwin_hall_3(x::T) where T<:Real
	"""
	function irwin_hall_3(x::T) where T<:Real
	    if x < zero(T)
	        return zero(T)
	    elseif x < one(T)
	        return convert(T, 0.5) * x^2
	    elseif x < convert(T, 2)
	        return convert(T, 0.5) * (-2*x^2 + 6*x - 3)
	    elseif x <= convert(T, 3)
	        return convert(T, 0.5) * (3-x)^2
	    else
	        return zero(T)
	    end
	end
	
	f = periodic(x -> irwin_hall_3(x + 3/2), n)
	
	"""
	    P(π::Vector{T}) where T<:Real
	
	Construct an approximate permutation matrix from vector `π` using the basis function `f`.
	Each element `π[i]` represents the approximate position in the permutation.
	
	# Arguments
	- `π::Vector{T}`: Vector representing an approximate permutation
	
	# Returns
	- A matrix `P` where `P[i,j] = f(π[i] - j)`
	"""
	function P(π::Vector{T}) where T<:Real
		n = size(π, 1)
		[ f(π[i] - j) for i in 1:n, j in 1:n ]
	end
	
	"""
	    F(A::Matrix{T}, P::Matrix{T}, c::Vector{T}=zeros(T,size(A,1))) where T<:Real
	
	Calculate the objective function `F(A,P,c) = -tr(A * P * A' * P') + diag(c) * P`
	This measures the approximate symmetry of a graph with adjacency matrix `A`
	under the permutation `P`, with optional penalty `c`.
	
	# Arguments
	- `A::Matrix{T}`: Adjacency matrix (symmetric)
	- `P::Matrix{T}`: Permutation matrix
	- `c::Vector{T}=zeros(T,size(A,1))`: Optional vector of penalty coefficients, defaults to zeros
	
	# Returns
	- The objective function value
	"""
	function F(
		A::Matrix{T},
		P::Matrix{T};
		c::Vector{T}=zeros(T,size(A,1))
	) where T<:Real
	    return -tr(A * P * A' * P') + tr(diagm(c) * P)
	end
	
	"""
	    G(A::Matrix{T}, π::Vector{T}, c::Vector{T}=zeros(T,size(A,1))) where T<:Real
	
	Calculate the objective function `G(A,π,c) = F(A,P(π),c)`.
	
	# Arguments
	- `A::Matrix{T}`: Input matrix
	- `π::Vector{T}`: Vector representing an approximate permutation
	- `c::Vector{T}=zeros(T,size(A,1))`: Optional vector of penalty coefficients, defaults to zeros
	
	# Returns
	- The objective function value
	"""
	function G(
		A::Matrix{T},
		π::Vector{T};
		c::Vector{T}=zeros(T,size(A,1))
	) where T<:Real
	    P_matrix = P(π)
	    return F(A, P_matrix, c=c)
	end
	
end

# ╔═╡ 1188b287-7b3b-4f5e-a7e4-a5766df2a406
"""
    createOptimizationProblem(n, objective_function)

Create an optimization problem for the sorting network polytope with a given objective function.
Returns an Optimization.jl problem.
"""
function createOptimizationProblem(n, A)

	function periodic(f::Function, period::Integer)
		return x -> f(x + period) + f(x) + f(x - period)
	end
	
	function irwin_hall_3(x::T) where T<:Real
		if x < zero(T)
			return zero(T)
		elseif x < one(T)
			return convert(T, 0.5) * x^2
		elseif x < convert(T, 2)
			return convert(T, 0.5) * (-2*x^2 + 6*x - 3)
		elseif x <= convert(T, 3)
			return convert(T, 0.5) * (3-x)^2
		else
			return zero(T)
		end
	end
	
	f = periodic(x -> irwin_hall_3(x + 3/2), n)
	
	function P(π::Vector)
		n = size(π, 1)
		[ f(π[i] - j) for i in 1:n, j in 1:n ]
	end
	

	function F(
		A::Matrix,
		P::Matrix;
		c::Vector=zeros(size(A,1))
	)
		return -tr(A * P * A' * P') + tr(diagm(c) * P)
	end

	function G(
		A::Matrix,
		π::Vector;
		c::Vector=zeros(size(A,1))
	)
		P_matrix = P(π)
		return F(A, P_matrix, c=c)
	end

	function objective_function(x, p)
		(A,) = p
		π = x[1:n]
		print(x)
		G(A, π)
	end
	
    num_vars, comparator_constraints, comparators = sortingNetworkPolytope(n)
	
	# Define your problem
	optf = OptimizationFunction(
		objective_function,
		Optimization.AutoForwardDiff(),
	)

	x0 = generateInitialPoint(n, comparators)
	p = (A,)
	prob = OptimizationProblem(
		optf, x0, p,
		equality_A = comparator_constraints.equality_A,
		equality_b = comparator_constraints.equality_b,
		inequality_A = comparator_constraints.inequality_A,
		inequality_b = comparator_constraints.inequality_b
	)

    return prob
end

# ╔═╡ de221157-e78d-42fe-8ba0-71b35866b449
# solve(prob, Ipopt.Optimizer())

# ╔═╡ 2a05f73a-b2e4-4046-930f-c76310e66ed3
num_vars, constraints, _ = sortingNetworkPolytope(10)

# ╔═╡ ebc92a35-94bd-43ca-bddf-e0ab105d0f80
x0 = generateInitialPoint(10, network)

# ╔═╡ 2eab7f51-348c-4a1e-b7d3-07957cb3f9ae
constraints.equality_A * x0 == constraints.equality_b

# ╔═╡ 6fdb68db-7bdd-4cfc-89d9-59e036e37c00
constraints.inequality_A * x0 <= constraints.inequality_b

# ╔═╡ dbd462cc-8e9e-4335-a004-778f017191ad
begin
	instance = Instance(
		"ER",
		"data/pidnebesna/ER/ER_nNodes20_density40.npz",
		20
	)
	A = load_instance_data(instance)["1"]

	prob = createOptimizationProblem(instance.n, A)

	solve(prob, Ipopt.Optimizer())
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Ipopt = "b6b21f68-93f8-5de0-b562-5493be1d77c9"
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
NPZ = "15e1cf62-19b3-5cfa-8e77-841668bca605"
Optimization = "7f7a1694-90dd-40f0-9382-eb1efda571ba"
OptimizationMOI = "fd9f6733-72f4-499f-8506-86b2bdd0dea1"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[compat]
Ipopt = "~1.7.2"
JuMP = "~1.24.0"
NPZ = "~0.4.3"
Optimization = "~3.19.3"
OptimizationMOI = "~0.1.12"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.3"
manifest_format = "2.0"
project_hash = "5a0a3960b69c3b98162ba3d02b94c06fdb75a4d3"

[[deps.ADTypes]]
git-tree-sha1 = "016833eb52ba2d6bea9fcb50ca295980e728ee24"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "0.2.7"

[[deps.ASL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6252039f98492252f9e47c312c8ffda0e3b9e78d"
uuid = "ae81ac8f-d209-56e5-92de-9978fef736f9"
version = "0.1.3+0"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "3b86719127f50670efe356bc11073d84b4ed7a5d"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.42"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "cd8b948862abee8f3d3e9b73a102a9ca924debb0"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.2.0"

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

    [deps.Adapt.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "017fcb757f8e921fb44ee063a7aafe5f89b86dd1"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.18.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "e38fbc49a620f5d0b660d7f543db1009fe0f8336"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.6.0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "TranscodingStreams"]
git-tree-sha1 = "84990fa864b7f2b4901901ca12736e45ee79068c"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.8.5"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConsoleProgressMonitor]]
deps = ["Logging", "ProgressMeter"]
git-tree-sha1 = "3ab7b2136722890b9af903859afcf457fa3059e8"
uuid = "88cd18e8-d9cc-4ea6-8889-5259c0d15c8b"
version = "0.1.2"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.ExproniconLite]]
git-tree-sha1 = "c13f0b150373771b0fdc1713c97860f8df12e6c2"
uuid = "55351af7-c7e9-48d6-89ff-24e801d99491"
version = "0.10.14"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "b66970a70db13f45b7e57fbda1736e1cf72174ea"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.17.0"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "a2df1b776752e3f344e5116c06d75a10436ab853"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.38"

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

    [deps.ForwardDiff.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers"]
git-tree-sha1 = "b104d487b34566608f8b4e1c39fb0b10aa279ff8"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "0.1.3"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.Ipopt]]
deps = ["Ipopt_jll", "LinearAlgebra", "MathOptInterface", "OpenBLAS32_jll", "PrecompileTools"]
git-tree-sha1 = "edcc4d1ec92e1abb5ebadc81b306c69ed58f53e2"
uuid = "b6b21f68-93f8-5de0-b562-5493be1d77c9"
version = "1.7.2"

[[deps.Ipopt_jll]]
deps = ["ASL_jll", "Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "MUMPS_seq_jll", "OpenBLAS32_jll", "Pkg"]
git-tree-sha1 = "e3e202237d93f18856b6ff1016166b0f172a49a8"
uuid = "9cc047cb-c261-5740-88fc-0cf96f7bdcc7"
version = "300.1400.400+0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "PrecompileTools", "StructTypes", "UUIDs"]
git-tree-sha1 = "1d322381ef7b087548321d3f878cb4c9bd8f8f9b"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.14.1"

    [deps.JSON3.extensions]
    JSON3ArrowExt = ["ArrowTypes"]

    [deps.JSON3.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.Jieko]]
deps = ["ExproniconLite"]
git-tree-sha1 = "2f05ed29618da60c06a87e9c033982d4f71d0b6c"
uuid = "ae98c720-c025-4a4a-838c-29b094483192"
version = "0.2.1"

[[deps.JuMP]]
deps = ["LinearAlgebra", "MacroTools", "MathOptInterface", "MutableArithmetics", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays"]
git-tree-sha1 = "cf832644f225dbe721bb9b97bf432007765fc695"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "1.24.0"

    [deps.JuMP.extensions]
    JuMPDimensionalDataExt = "DimensionalData"

    [deps.JuMP.weakdeps]
    DimensionalData = "0703355e-b756-11e9-17c0-8b28908087d0"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "fb6803dafae4a5d62ea5cab204b1e657d9737e7f"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.2.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.METIS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2eefa8baa858871ae7770c98c3c2a7e46daba5b4"
uuid = "d00139f3-1899-568f-a2f0-47f597d42d70"
version = "5.1.3+0"

[[deps.MUMPS_seq_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "METIS_jll", "OpenBLAS32_jll", "Pkg"]
git-tree-sha1 = "29de2841fa5aefe615dea179fcde48bb87b58f57"
uuid = "d7ed1dd3-d0ae-5e8e-bfb4-87a502085b8d"
version = "5.4.1+0"

[[deps.MacroTools]]
git-tree-sha1 = "72aebe0b5051e5143a079a4685a46da330a40472"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.15"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "DataStructures", "ForwardDiff", "JSON3", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays", "SpecialFunctions", "Test"]
git-tree-sha1 = "b691a4b4c8ef7a4fba051d546040bfd2ae6f0719"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.37.2"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.Moshi]]
deps = ["ExproniconLite", "Jieko"]
git-tree-sha1 = "453de0fc2be3d11b9b93ca4d0fddd91196dcf1ed"
uuid = "2e0e35c7-a2e4-4343-998d-7ef72827ed2d"
version = "0.3.5"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "491bdcdc943fcbc4c005900d7463c9f216aabf4c"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.6.4"

[[deps.NPZ]]
deps = ["FileIO", "ZipFile"]
git-tree-sha1 = "60a8e272fe0c5079363b28b0953831e2dd7b7e6f"
uuid = "15e1cf62-19b3-5cfa-8e77-841668bca605"
version = "0.4.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "cc0a5deefdb12ab3a096f00a6d42133af4560d71"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS32_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ece4587683695fe4c5f20e990da0ed7e83c351e7"
uuid = "656ef2d0-ae68-5445-9ca0-591084a874a2"
version = "0.3.29+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Optimization]]
deps = ["ADTypes", "ArrayInterface", "ConsoleProgressMonitor", "DocStringExtensions", "LinearAlgebra", "Logging", "LoggingExtras", "Pkg", "Printf", "ProgressLogging", "Reexport", "Requires", "SciMLBase", "SparseArrays", "TerminalLoggers"]
git-tree-sha1 = "1aa7ffea6e171167e9cae620d749e16d5874414a"
uuid = "7f7a1694-90dd-40f0-9382-eb1efda571ba"
version = "3.19.3"

    [deps.Optimization.extensions]
    OptimizationEnzymeExt = "Enzyme"
    OptimizationFiniteDiffExt = "FiniteDiff"
    OptimizationForwardDiffExt = "ForwardDiff"
    OptimizationMTKExt = "ModelingToolkit"
    OptimizationReverseDiffExt = "ReverseDiff"
    OptimizationSparseDiffExt = ["SparseDiffTools", "Symbolics", "ReverseDiff"]
    OptimizationTrackerExt = "Tracker"
    OptimizationZygoteExt = "Zygote"

    [deps.Optimization.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    ModelingToolkit = "961ee093-0014-501f-94e3-6117800e7a78"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseDiffTools = "47a9eef4-7e08-11e9-0b38-333d64bd3804"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.OptimizationMOI]]
deps = ["Ipopt_jll", "MathOptInterface", "Optimization", "Reexport", "SparseArrays"]
git-tree-sha1 = "6078c5864fc3ab80e7fba1e0ea2258ac23fef56d"
uuid = "fd9f6733-72f4-499f-8506-86b2bdd0dea1"
version = "0.1.12"

[[deps.OrderedCollections]]
git-tree-sha1 = "cc4054e898b852042d7b503313f7ad03de99c3dd"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "8f6bc219586aef8baf0ff9a5fe16ee9c70cb65e4"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.2"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "DocStringExtensions", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables"]
git-tree-sha1 = "e96b644f7bfbf1015f8e42a7c7abfae2a48fafbf"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "3.31.0"

    [deps.RecursiveArrayTools.extensions]
    RecursiveArrayToolsFastBroadcastExt = "FastBroadcast"
    RecursiveArrayToolsForwardDiffExt = "ForwardDiff"
    RecursiveArrayToolsMeasurementsExt = "Measurements"
    RecursiveArrayToolsMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    RecursiveArrayToolsReverseDiffExt = ["ReverseDiff", "Zygote"]
    RecursiveArrayToolsSparseArraysExt = ["SparseArrays"]
    RecursiveArrayToolsStructArraysExt = "StructArrays"
    RecursiveArrayToolsTrackerExt = "Tracker"
    RecursiveArrayToolsZygoteExt = "Zygote"

    [deps.RecursiveArrayTools.weakdeps]
    FastBroadcast = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "04c968137612c4a5629fa531334bb81ad5680f00"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.13"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SciMLBase]]
deps = ["ADTypes", "Accessors", "ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "Moshi", "PrecompileTools", "Preferences", "Printf", "RecipesBase", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLOperators", "SciMLStructures", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface"]
git-tree-sha1 = "ee305515b0946db5f56af699e8b5804fee04146c"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "2.75.1"

    [deps.SciMLBase.extensions]
    SciMLBaseChainRulesCoreExt = "ChainRulesCore"
    SciMLBaseMLStyleExt = "MLStyle"
    SciMLBaseMakieExt = "Makie"
    SciMLBasePartialFunctionsExt = "PartialFunctions"
    SciMLBasePyCallExt = "PyCall"
    SciMLBasePythonCallExt = "PythonCall"
    SciMLBaseRCallExt = "RCall"
    SciMLBaseZygoteExt = "Zygote"

    [deps.SciMLBase.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    MLStyle = "d8e11817-5142-5d16-987a-aa16d5891078"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    PartialFunctions = "570af359-4316-4cb7-8c74-252c00c2016b"
    PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.SciMLOperators]]
deps = ["Accessors", "ArrayInterface", "DocStringExtensions", "LinearAlgebra", "MacroTools"]
git-tree-sha1 = "6149620767866d4b0f0f7028639b6e661b6a1e44"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "0.3.12"
weakdeps = ["SparseArrays", "StaticArraysCore"]

    [deps.SciMLOperators.extensions]
    SciMLOperatorsSparseArraysExt = "SparseArrays"
    SciMLOperatorsStaticArraysCoreExt = "StaticArraysCore"

[[deps.SciMLStructures]]
deps = ["ArrayInterface"]
git-tree-sha1 = "566c4ed301ccb2a44cbd5a27da5f885e0ed1d5df"
uuid = "53ae85a6-f571-4167-b2af-e1d143709226"
version = "1.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "64cca0c26b4f31ba18f13f6c12af7c85f478cfde"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.0"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "159331b30e94d7b11379037feeb9b690950cace8"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.SymbolicIndexingInterface]]
deps = ["Accessors", "ArrayInterface", "RuntimeGeneratedFunctions", "StaticArraysCore"]
git-tree-sha1 = "d6c04e26aa1c8f7d144e1a8c47f1c73d3013e289"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.3.38"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "f133fab380933d042f6796eda4e130272ba520ca"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.7"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "f492b7fe1698e623024e873244f10d89c95c340a"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.10.1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╟─a69c0df2-fce7-11ef-14c8-ad51bfb0e79d
# ╠═4b346729-aafc-40c7-af93-2073ee7e2dcf
# ╠═fb33e25d-f5c6-4faf-8884-d853aca3b1a1
# ╠═894e8680-a849-45dd-8edb-91136986b256
# ╠═e8589476-2744-42d2-8ef5-36f705c84a1e
# ╠═74d2e80d-f9f4-4bea-8643-93f5934eed01
# ╠═82c3000e-4ae2-44a5-95e7-95e8c90fa4cc
# ╠═97ba189f-5d3d-406d-bc19-06059810a950
# ╠═03c283a3-e26e-4e64-b0ca-c315cf2fa8e8
# ╠═af3bffaa-3e93-4a76-8f35-31122920ca53
# ╠═747bfc6e-4d9d-48cd-9d13-ae272ae5d460
# ╠═1188b287-7b3b-4f5e-a7e4-a5766df2a406
# ╠═de221157-e78d-42fe-8ba0-71b35866b449
# ╠═2a05f73a-b2e4-4046-930f-c76310e66ed3
# ╠═ebc92a35-94bd-43ca-bddf-e0ab105d0f80
# ╠═2eab7f51-348c-4a1e-b7d3-07957cb3f9ae
# ╠═6fdb68db-7bdd-4cfc-89d9-59e036e37c00
# ╠═dbd462cc-8e9e-4335-a004-778f017191ad
# ╠═faf40b51-c850-430c-bf59-eb99da68631a
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
