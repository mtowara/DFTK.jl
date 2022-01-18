# Contains the physical specification of the model

# A physical specification of a model.
# Contains the geometry information, but no discretization parameters.
# The exact model used is defined by the list of terms.
struct Model{T <: Real}
    # Human-readable name for the model (like LDA, PBE, ...)
    model_name::String

    # AtomsBase-compatible description of the physical system
    system::AbstractSystem

    # Lattice and reciprocal lattice vectors in columns
    lattice::Mat3{T}
    recip_lattice::Mat3{T}
    unit_cell_volume::T
    recip_cell_volume::T
    n_dim::Int  # Dimension of the system; 3 unless `lattice` has zero columns

    # Electrons, occupation and smearing function
    n_electrons::Int  # usually consistent with `atoms` field, but doesn't have to

    # spin_polarization values:
    #     :none       No spin polarization, αα and ββ density identical,
    #                 αβ and βα blocks zero, 1 spin component treated explicitly
    #     :collinear  Spin is polarized, but everywhere in the same direction.
    #                 αα ̸= ββ, αβ = βα = 0, 2 spin components treated
    #     :full       Generic magnetization, non-uniform direction.
    #                 αβ, βα, αα, ββ all nonzero, different (not supported)
    #     :spinless   No spin at all ("spinless fermions", "mathematicians' electrons").
    #                 The difference with :none is that the occupations are 1 instead of 2
    spin_polarization::Symbol
    n_spin_components::Int  # 2 if :collinear, 1 otherwise

    # If temperature==0, no fractional occupations are used.
    # If temperature is nonzero, the occupations are
    # `fn = max_occ*smearing((εn-εF) / temperature)`
    temperature::T
    smearing::Smearing.SmearingFunction # see Smearing.jl for choices

    # Vector of Pairs Element => Vec3 (positions, fractional coordinates)
    # Possibly empty. `atoms` just contain the information on the atoms, it's up to
    # the `terms` to make use of it (or not)
    atoms::Vector{Pair{Any, Vec3{T}}}

    # Groups of identical atomic potentials (to speed up term instantiation)
    potential_groups::Vector{Vector{Int}}

    # Deprecated field, contained for compatibility. Will be dropped in favour of `atoms`.
    oldatoms::Vector{Pair{Any, Vector{Vec3{T}}}}

    # each element t must implement t(basis), which instantiates a
    # term in a given basis and gives back a term (<: Term)
    # see terms.jl for some default terms
    term_types::Vector

    # list of symmetries of the model
    symmetries::Vector{SymOp}
end

# TODO To ensure code breaks if "old" atoms are used
AtomsType = AbstractVector{<:Pair{<:Any,<:AbstractVector{<:Real}}}

# TODO For compatibility (because I can't be bothered to change everything right now)
function oldatoms_from_new(atomic_potentials, payload)
    potentials = first.(atomic_potentials)
    potential_groups = [findall(Ref(pot) .== potentials) for pot in Set(potentials)]
    new_potentials = [
        first(atomic_potentials[first(group)]) => last.(atomic_potentials[group])
         for group in potential_groups
    ]
    isempty(payload) && return new_potentials, payload

    new_payload = [first(atomic_potentials[first(group)]) => payload[group]
                   for group in potential_groups]

    new_potentials, new_payload
end
oldatoms_from_new(at) = first(oldatoms_from_new(at, at))

old_to_new(atoms::AtomsType) = atoms
function old_to_new(atoms::AbstractVector)
    [el => pos for (el, positions) in atoms for pos in positions]
end
# end for compatibility


"""
    Model(lattice; n_electrons, atoms, magnetic_moments, terms,
                   temperature, smearing, spin_polarization, symmetry)

Creates the physical specification of a model (without any
discretization information).

`n_electrons` is taken from `atoms` if not specified

`spin_polarization` is :none by default (paired electrons)
unless any of the elements has a non-zero initial magnetic moment.
In this case the spin_polarization will be :collinear.

`magnetic_moments` is only used to determine the symmetry and the
`spin_polarization`; it is not stored inside the datastructure.

`smearing` is Fermi-Dirac if `temperature` is non-zero, none otherwise

The `symmetries` kwarg allows (a) to pass `true` / `false` to enable / disable
the automatic determination of lattice symmetries or (b) to pass an explicit list
of symmetry operations to use for lowering the computational effort.
The default behaviour is equal to `true`, namely that the code checks the
specified model in form of the Hamiltonian `terms`, `lattice`, `atoms` and
`magnetic_moments` parameters and from these automatically determines a set of symmetries
it can safely use. If you want to pass custom symmetry operations (e.g. a reduced or
extended set) use the `symmetry_operations` function. Notice that this may lead to wrong
results if e.g. the external potential breaks some of the passed symmetries. Use `false` to
turn off symmetries completely.
"""
function Model(lattice::AbstractMatrix{T};
               model_name="custom",
               system=nothing,
               n_electrons=nothing,
               atoms::AtomsType=Pair{Any,Vector{Float64}}[],
               magnetic_moments=[],
               terms=[Kinetic()],
               temperature=T(0.0),
               smearing=nothing,
               spin_polarization=default_spin_polarization(magnetic_moments),
               symmetries=default_symmetries(lattice, atoms, magnetic_moments,
                                             spin_polarization, terms),
               ) where {T <: Real}
    lattice = Mat3{T}(lattice)
    temperature = T(austrip(temperature))

    if isnothing(n_electrons)
        # Get it from the atomic potentials, assuming charge-neutral cell
        isempty(atoms) && error(
            "Either n_electrons or a non-empty atoms should be provided."
        )
        n_electrons = sum(n_elec_valence(potential) for (potential, _) in atoms)
    else
        @assert n_electrons isa Int
    end
    isempty(terms) && error("Model without terms not supported.")
    isnothing(system) && (system = periodic_system(Atom[], collect(eachcol(lattice)) * u"bohr"))

    # Special handling of 1D and 2D systems, and sanity checks
    n_dim = count(!iszero, eachcol(lattice))
    n_dim > 0 || error("Check your lattice; we do not do 0D systems")
    for i = n_dim+1:3
        norm(lattice[:, i]) == norm(lattice[i, :]) == 0 || error(
            "For 1D and 2D systems, the non-empty dimensions must come first")
    end
    _is_well_conditioned(lattice[1:n_dim, 1:n_dim]) || @warn (
        "Your lattice is badly conditioned, the computation is likely to fail.")

    # Note: In the 1D or 2D case, the volume is the length/surface
    recip_lattice = compute_recip_lattice(lattice)
    unit_cell_volume  = compute_unit_cell_volume(lattice)
    recip_cell_volume = compute_unit_cell_volume(recip_lattice)

    spin_polarization in (:none, :collinear, :full, :spinless) ||
        error("Only :none, :collinear, :full and :spinless allowed for spin_polarization")
    spin_polarization == :full && error("Full spin polarization not yet supported")
    !isempty(magnetic_moments) && !(spin_polarization in (:collinear, :full)) && @warn(
        "Non-empty magnetic_moments on a Model without spin polarization detected."
    )
    n_spin = length(spin_components(spin_polarization))

    if smearing === nothing
        @assert temperature >= 0
        # Default to Fermi-Dirac smearing when finite temperature
        smearing = temperature > 0.0 ? Smearing.FermiDirac() : Smearing.None()
    end

    if !allunique(string.(nameof.(typeof.(terms))))
        error("Having several terms of the same name is not supported.")
    end

    # Determine symmetry operations to use
    symmetries == true  && (symmetries = default_symmetries(lattice, atoms, magnetic_moments,
                                                            spin_polarization, terms))
    symmetries == false && (symmetries = [identity_symop()])
    @assert !isempty(symmetries)  # Identity has to be always present.

    # Determine the groups of unique atomic potentials
    potentials = first.(atoms)
    potential_groups = [Ref(pot) .== potentials for pot in Set(potentials)]

    Model{T}(model_name, system, lattice, recip_lattice, unit_cell_volume, recip_cell_volume,
             n_dim, n_electrons, spin_polarization, n_spin, T(temperature), smearing,
             atoms, potential_groups, oldatoms_from_new(atoms), terms, symmetries)
end
Model(lattice::AbstractMatrix{T}; kwargs...) where {T <: Integer}  = Model(Float64.(lattice); kwargs...)
Model(lattice::AbstractMatrix{Q}; kwargs...) where {Q <: Quantity} = Model(austrip.(lattice); kwargs...)

function Model(system::AbstractSystem; kwargs...)
    @assert !(:system in keys(kwargs))
    @assert !(:atoms  in keys(kwargs))
    parsed = parse_system(system)
    Model(parsed.lattice; atoms=parsed.atoms, parsed.kwargs..., kwargs...)
end


normalize_magnetic_moment(::Nothing)  = Vec3{Float64}(zeros(3))
normalize_magnetic_moment(mm::Number) = Vec3{Float64}(0, 0, mm)
normalize_magnetic_moment(mm::AbstractVector) = Vec3{Float64}(mm)

"""
:none if no element has a magnetic moment, else :collinear or :full
"""
function default_spin_polarization(magnetic_moments)
    isempty(magnetic_moments) && return :none
    all_magmoms = normalize_magnetic_moment.(magnetic_moments)
    all(iszero, all_magmoms) && return :none
    all(iszero(magmom[1:2]) for magmom in all_magmoms) && return :collinear

    :full
end

"""
Default logic to determine the symmetry operations to be used in the model.
"""
function default_symmetries(lattice, atoms, magnetic_moments, spin_polarization, terms;
                            tol_symmetry=1e-5)
    dimension = count(!iszero, eachcol(lattice))
    if spin_polarization == :full || dimension != 3
        return [identity_symop()]  # Symmetry not supported in spglib
    elseif spin_polarization == :collinear && isempty(magnetic_moments)
        # Spin-breaking due to initial magnetic moments cannot be determined
        return [identity_symop()]
    elseif any(breaks_symmetries, terms)
        return [identity_symop()]  # Terms break symmetry
    else
        magmoms = normalize_magnetic_moment.(magnetic_moments)
        oldatoms, oldmagmoms = oldatoms_from_new(atoms, magmoms)
        return symmetry_operations(lattice, oldatoms, oldmagmoms,
                                   tol_symmetry=tol_symmetry)
    end
end


function parse_system(system::AbstractSystem{D}) where {D}
    if !all(periodicity(system))
        error("DFTK only supports calculations with periodic boundary conditions.")
    end

    # Parse abstract system and return data required to construct model
    mtx = austrip.(hcat(bounding_box(system)...))
    T = eltype(mtx)
    lattice = zeros(T, 3, 3)
    lattice[1:D, 1:D] .= mtx

    # Cache for instantiated pseudopotentials (such that the respective objects are
    # indistinguishable in memory. We need that property to fill potential_groups in Model)
    cached_pseudos = Dict{String,Any}()
    atoms = map(system) do atom
        if hasproperty(atom, :potential)
            potential = atom.potential
        elseif hasproperty(atom, :pseudopotential)
            pspkey = atom.pseudopotential
            if !(pspkey in keys(cached_pseudos))
                cached_pseudos[pspkey] = ElementPsp(atomic_symbol(atom); psp=load_psp(pspkey))
            end
            potential = cached_pseudos[pspkey]
        else
            potential = ElementCoulomb(atomic_symbol(atom))
        end

        coordinate = zeros(T, 3)
        coordinate[1:D] = lattice[1:D, 1:D] \ T.(austrip.(position(atom)))
        potential => Vec3{T}(coordinate)
    end

    magnetic_moments = map(system) do atom
        hasproperty(atom, :magnetic_moment) || return nothing
        getproperty(atom, :magnetic_moment)
    end
    if all(m -> isnothing(m) || iszero(m) || isempty(m), magnetic_moments)
        empty!(magnetic_moments)
    else
        magnetic_moments = normalize_magnetic_moment.(magnetic_moments)
    end

    # TODO Use system to determine n_electrons

    (; lattice, atoms, kwargs=(; system, magnetic_moments))
end


"""
Maximal occupation of a state (2 for non-spin-polarized electrons, 1 otherwise).
"""
function filled_occupation(model)
    if model.spin_polarization in (:spinless, :collinear)
        return 1
    elseif model.spin_polarization == :none
        return 2
    else
        error("Not implemented $(model.spin_polarization)")
    end
end


"""
Explicit spin components of the KS orbitals and the density
"""
function spin_components(spin_polarization::Symbol)
    spin_polarization == :collinear && return (:up, :down  )
    spin_polarization == :none      && return (:both,      )
    spin_polarization == :spinless  && return (:spinless,  )
    spin_polarization == :full      && return (:undefined, )
end
spin_components(model::Model) = spin_components(model.spin_polarization)


_is_well_conditioned(A; tol=1e5) = (cond(A) <= tol)
