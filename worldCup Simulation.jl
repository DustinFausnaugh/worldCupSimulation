using LinearAlgebra, Printf, SparseArrays

# 1. Define the match structure
struct WorldCup
    homeTeam::String
    awayTeam::String
    home_goals::Int
    away_goals::Int
end

# 2. List of actual teams in the World Cup
teams = [
    "Brazil", "United States", "Algeria", "Argentina","Australia","Austria",
    "Belgium", "Bosnia", "Cape verde", "Columbia", "Congo DR",
    "Cote d'Ivoire", "Croatia", "Curacao", "Chechia", "Ecuador", "Japan",
    "Egypt", "England", "France", "Germany", "Ghana", "Haiti","Iran", "Iraq",
    "Jordan", "South Korea", "Morocco", "Netherlands", "New Zealand", "Norway",
    "Panama", "Paraguay", "portugal", "Qatar", "Saudi Arabia", "Scotland",
    "sengal", "South Africa", "Spain", "Sweden", "Switzerland", "Tunisia",
    "Turkiye", "Uruguay", "Uzbekistan", "Mexico", "Canada"
]

# 3. Creating the matches for the World Cup
matches = [
    WorldCup("Mexico", "South Africa", 2, 1),
    WorldCup("South Korea", "Chechia", 1, 1),
    WorldCup("Canada", "Bosnia", 2, 2),
    WorldCup("Qatar", "Switzerland", 3, 1),
    WorldCup("Brazil", "Morocco", 1, 1),
    WorldCup("Haiti", "Scotland", 2, 0),
    WorldCup("United States", "Paraguay", 1, 4),
    WorldCup("Australia", "Turkiye", 1, 1),
    WorldCup("Germany", "Curacao", 1, 3),
    WorldCup("Cote d'Ivoire", "Ecuador", 2, 2),
    WorldCup("Netherlands", "Japan", 3, 3),
    WorldCup("Sweden", "Tunisia", 3, 2),
    WorldCup("Belgium", "Egypt", 1, 4),
    WorldCup("Iran", "New Zealand", 5, 1),
    WorldCup("Spain", "Cape verde", 3, 2),
    WorldCup("Saudi Arabia", "Uruguay", 2, 2),
    WorldCup("France", "sengal", 1, 1),
    WorldCup("Iraq", "Norway", 1, 1),
    WorldCup("Argentina", "Algeria", 4, 0),
    WorldCup("Austria", "Jordan", 0, 0),
    WorldCup("portugal", "Congo DR", 1, 1),
    WorldCup("Uzbekistan", "Columbia", 1, 0),
    WorldCup("England", "Croatia", 0, 1),
    WorldCup("Ghana", "Panama", 2, 2)
]

#Build the dictionary to map names to IDs
teamDictionary = Dict{String, Int}()
for (index, teamName) in enumerate(teams)
    teamDictionary[teamName] = index
end

# Constants (No intercept column needed)
num_teams = length(teams)
total_columns = 2 * num_teams

# Initialize tracking variables and arrays
current_row = 1
I = Int[]
J = Int[]
V = Float64[]
y_raw = Float64[]

# Loop through matches to populate your design matrix coordinates
for match in matches
    global current_row
    
    home_id = teamDictionary[match.homeTeam]
    away_id = teamDictionary[match.awayTeam]
    
    # Showing how home team scores
    append!(I, [current_row, current_row]) 
    append!(J, [home_id, away_id + num_teams])
    append!(V, [1.0, 1.0])
    push!(y_raw, log(match.home_goals + 0.1))
    current_row += 1

    # Away team scores
    append!(I, [current_row, current_row]) 
    append!(J, [away_id, home_id + num_teams])
    append!(V, [1.0, 1.0])
    push!(y_raw, log(match.away_goals + 0.1))
    current_row += 1
end

# Construct the rectangular design matrix A
total_rows = current_row - 1
A_sparse = sparse(I, J, V, total_rows, total_columns)
A = Matrix(A_sparse)

# Center the target vector y by subtracting the global mean
baseline_mean = sum(y_raw) / length(y_raw)
y = y_raw .- baseline_mean



# Form the Normal Equations: (A' * A) * x = A' * y
S_raw = A' * A
b = A' * y

# Add ridge regularization to handle the underdetermined system cleanly
lambda = 0.1
S = S_raw + lambda * LinearAlgebra.I

println("Successfully formed square Normal Equations matrix S of size: ", size(S))

# Perform LUP Decomposition (S = P' * L * U)
F = lu(S)
coefficients = F \ b
println("System solved exactly using LUP Decomposition!")

function simulate_match(home_team::String, away_team::String, team_dict, coeffs, num_teams, baseline)
    home_idx = team_dict[home_team]
    away_idx = team_dict[away_team]
    
    home_attack = coeffs[home_idx]
    home_defense = coeffs[home_idx + num_teams]
    away_attack = coeffs[away_idx]
    away_defense = coeffs[away_idx + num_teams]
    
    expected_home_log = baseline + home_attack + away_defense
    expected_away_log = baseline + away_attack + home_defense
    
    expected_home_goals = max(0.0, exp(expected_home_log) - 0.1)
    expected_away_goals = max(0.0, exp(expected_away_log) - 0.1)
    
    sim_home_score = round(Int, expected_home_goals)
    sim_away_score = round(Int, expected_away_goals)
    
    println("\n SIMULATED MATCH RESULT:")
    @printf("%s (%d) vs %s (%d)\n", home_team, sim_home_score, away_team, sim_away_score)
    @printf("  [Expected Stats -> %s: %.2f goals | %s: %.2f goals]\n", 
            home_team, expected_home_goals, away_team, expected_away_goals)
end

# Extract and display the ratings for each team
println("\n=== WORLD CUP TEAM RATINGS (LUP SOLVER) ===")
printf_format = "%-20s | Attack: %6.3f | Defense: %6.3f\n"

println("Baseline Tournament Intercept (Log-Goals): ", round(baseline_mean, digits=3))
println("-"^55)

for (index, team_name) in enumerate(teams)
    attack_rating = coefficients[index]
    defense_rating = coefficients[index + num_teams]
    
    @printf("%-20s | Attack: %6.3f | Defense: %6.3f\n", team_name, attack_rating, defense_rating)
end
function simulate_tournament_match(home_team::String, away_team::String, team_dict, coeffs, num_teams, baseline)
    home_idx = team_dict[home_team]
    away_idx = team_dict[away_team]
    
    # Calculate expected goals
    expected_home_log = baseline + coeffs[home_idx] + coeffs[away_idx + num_teams]
    expected_away_log = baseline + coeffs[away_idx] + coeffs[home_idx + num_teams]
    
    expected_home_goals = max(0.0, exp(expected_home_log) - 0.1)
    expected_away_goals = max(0.0, exp(expected_away_log) - 0.1)
    
    sim_home_score = round(Int, expected_home_goals)
    sim_away_score = round(Int, expected_away_goals)
    
    # Handle penalty shootout if there's a draw in the knockout stage
    if sim_home_score == sim_away_score
        # Predict winner based on fractional expected goal advantage
        return expected_home_goals >= expected_away_goals ? home_team : away_team
    else
        return sim_home_score > sim_away_score ? home_team : away_team
    end
end

println("\n=======================================================")
println(" RUNNING COMPLETE WORLD CUP KNOCKOUT BRACKET ")
println("=======================================================")

# Dynamically calculate a "Power Rating" (Attack minus Defense) for every team
# Lower defense rating is better, so subtracting it makes a stronger positive rating
team_power = []
for (index, team_name) in enumerate(teams)
    power = coefficients[index] - coefficients[index + num_teams]
    push!(team_power, (team_name, power))
end

# Sort the teams by power rating and select the Top 16
sort!(team_power, by = x -> x[2], rev = true)
qualified_teams = [team_power[i][1] for i in 1:16]

println("Top 16 Teams Qualified Based on LUP Matrix Solving:")
for i in 1:16
    @printf("  %d. %-15s (Power: %.3f)\n", i, team_power[i][1], team_power[i][2])
end
println("-"^55)

# Run the Knockout Rounds sequentially
round_teams = copy(qualified_teams)
round_names = ["ROUND OF 16", "QUARTERFINALS", "SEMIFINALS", "WORLD CUP FINAL"]

for round_name in round_names
    println("\n --- $round_name --- ")
    next_round_teams = String[]
    
    # Pair teams up (1st vs 16th, 2nd vs 15th style seeds)
    num_matches = div(length(round_teams), 2)
    for i in 1:num_matches
        home = round_teams[i]
        away = round_teams[end - i + 1]
        
        winner = simulate_tournament_match(home, away, teamDictionary, coefficients, num_teams, baseline_mean)
        push!(next_round_teams, winner)
        
        println("  $home vs $away -> WINNER: $winner")
    end
    
    # Update the remaining pool of teams for the next round
    global round_teams = next_round_teams
    
    if length(round_teams) == 1
        println("\n===================================================")
        println("  WORLD CUP CHAMPION DIRECTLY PREDICTED BY LUP: $(round_teams[1]) ")
        println("===================================================")
        break
    end
end