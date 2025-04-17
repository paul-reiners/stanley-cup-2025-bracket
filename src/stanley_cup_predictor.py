import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Additional imports for player data
from concurrent.futures import ThreadPoolExecutor
import time


class StanleyCupPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.teams_data = {}
        self.features = None
        self.target = None

        # NHL API base URLs
        self.nhl_api_url = "https://api.nhle.com/stats/rest/en"
        self.nhl_web_api_url = "https://api-web.nhle.com"

        # For rate limiting API requests
        self.request_delay = 0.5  # seconds between requests to avoid hitting rate limits

    def fetch_historical_data(self, start_season=2010, end_season=2024):
        """
        Fetch historical NHL data from the NHL API

        Parameters:
        - start_season: First season to include (start year)
        - end_season: Last season to include (end year)

        Returns:
        - DataFrame with historical team and playoff data
        """
        all_data = []

        for season in range(start_season, end_season + 1):
            season_str = f"{season}{season + 1}"

            # Use the NHL API endpoint for team stats
            # NHL API uses format 20232024 for the 2023-2024 season
            url = f"https://api.nhle.com/stats/rest/en/team/summary"
            params = {
                'sort': 'points',
                'cayenneExp': f'seasonId={season_str} and gameTypeId=2'  # Regular season games
            }

            try:
                # Fetch the actual data
                response = requests.get(url, params=params)

                # Check if request was successful
                if response.status_code == 200:
                    data = response.json()

                    # Process the data from the API
                    if 'data' in data:
                        for team_data in data['data']:
                            # Convert API data to our format
                            team_record = {
                                'season': season,
                                'team': team_data.get('teamFullName', ''),
                                'wins': team_data.get('wins', 0),
                                'losses': team_data.get('losses', 0),
                                'ot_losses': team_data.get('otLosses', 0),
                                'points': team_data.get('points', 0),
                                'goal_differential': team_data.get('goalDifferential', 0),
                                'powerplay_percentage': team_data.get('powerPlayPct', 0),
                                'penalty_kill_percentage': team_data.get('penaltyKillPct', 0)
                            }

                            # Add playoff data (need to fetch from another endpoint or determine from standings)
                            # For now, we'll estimate based on points (top 16 teams make playoffs)
                            team_record['made_playoffs'] = 0  # Will be set later
                            team_record['playoff_rounds'] = 0  # Will be set later
                            team_record['won_cup'] = 0  # Will be set later

                            all_data.append(team_record)
                else:
                    print(f"Error: API returned status code {response.status_code} for season {season_str}")
                    # Fall back to synthetic data if the API fails
                    fallback_data = self._generate_synthetic_data(season, 32)
                    all_data.extend(fallback_data)
                    print(f"Using synthetic data for season {season_str}")

            except Exception as e:
                print(f"Error fetching data for season {season_str}: {e}")
                # Fall back to synthetic data if the API fails
                fallback_data = self._generate_synthetic_data(season, 32)
                all_data.extend(fallback_data)
                print(f"Using synthetic data for season {season_str}")

        # Convert to DataFrame
        df = pd.DataFrame(all_data)

        # Determine playoff teams (top 16 in points each season)
        for season in range(start_season, end_season + 1):
            season_teams = df[df['season'] == season].sort_values(by='points', ascending=False)
            playoff_teams_indices = season_teams.head(16).index

            # Mark playoff teams
            df.loc[playoff_teams_indices, 'made_playoffs'] = 1

            # Simulate or fetch playoff rounds advancement
            # For demonstration, we'll simulate this based on regular season performance
            round_advancements = {
                'round1': playoff_teams_indices[:8],  # Top 8 advance to round 2
                'round2': playoff_teams_indices[:4],  # Top 4 advance to conf finals
                'round3': playoff_teams_indices[:2],  # Top 2 advance to finals
                'champion': playoff_teams_indices[:1]  # Champion
            }

            df.loc[round_advancements['round1'], 'playoff_rounds'] = 2
            df.loc[round_advancements['round2'], 'playoff_rounds'] = 3
            df.loc[round_advancements['round3'], 'playoff_rounds'] = 4
            df.loc[round_advancements['champion'], 'playoff_rounds'] = 5
            df.loc[round_advancements['champion'], 'won_cup'] = 1

        return df

    def _generate_synthetic_data(self, season, num_teams):
        """Generate synthetic data for demonstration purposes"""
        teams = [f"Team_{i}" for i in range(1, num_teams + 1)]
        data = []

        for team in teams:
            # Regular season stats
            wins = np.random.randint(20, 60)
            losses = np.random.randint(10, 40)
            ot_losses = np.random.randint(0, 15)
            points = wins * 2 + ot_losses

            # Advanced stats
            goal_diff = np.random.randint(-100, 150)
            pp_percentage = np.random.uniform(15, 30)
            pk_percentage = np.random.uniform(70, 90)

            # Create a record for if they made playoffs and how far they went
            made_playoffs = 1 if points > 90 else 0

            # Playoff rounds advancement (0=missed, 1=first round, 2=second, 3=conf finals, 4=finals, 5=champion)
            if made_playoffs:
                playoff_rounds = np.random.choice([1, 2, 3, 4, 5], p=[0.5, 0.25, 0.125, 0.075, 0.05])
            else:
                playoff_rounds = 0

            # If team wins Stanley Cup
            won_cup = 1 if playoff_rounds == 5 else 0

            data.append({
                'season': season,
                'team': team,
                'wins': wins,
                'losses': losses,
                'ot_losses': ot_losses,
                'points': points,
                'goal_differential': goal_diff,
                'powerplay_percentage': pp_percentage,
                'penalty_kill_percentage': pk_percentage,
                'made_playoffs': made_playoffs,
                'playoff_rounds': playoff_rounds,
                'won_cup': won_cup
            })

        return data

    def fetch_current_season_data(self, season=2024):
        """
        Fetch data for the current season to make predictions

        Parameters:
        - season: Current season (start year)

        Returns:
        - DataFrame with current season team data
        """
        season_str = f"{season}{season + 1}"

        # NHL API endpoint for current team stats
        url = f"https://api.nhle.com/stats/rest/en/team/summary"
        params = {
            'sort': 'points',
            'cayenneExp': f'seasonId={season_str} and gameTypeId=2'  # Regular season games
        }

        try:
            # Fetch the actual data
            response = requests.get(url, params=params)

            if response.status_code == 200:
                data = response.json()

                current_data = []
                if 'data' in data:
                    for team_data in data['data']:
                        team_record = {
                            'season': season,
                            'team': team_data.get('teamFullName', ''),
                            'wins': team_data.get('wins', 0),
                            'losses': team_data.get('losses', 0),
                            'ot_losses': team_data.get('otLosses', 0),
                            'points': team_data.get('points', 0),
                            'goal_differential': team_data.get('goalDifferential', 0),
                            'powerplay_percentage': team_data.get('powerPlayPct', 0),
                            'penalty_kill_percentage': team_data.get('penaltyKillPct', 0),
                            # Additional advanced stats if available
                            'shots_for_per_game': team_data.get('shotsForPerGame', 0),
                            'shots_against_per_game': team_data.get('shotsAgainstPerGame', 0),
                            'faceoff_win_percentage': team_data.get('faceoffWinPct', 0)
                        }
                        current_data.append(team_record)

                return pd.DataFrame(current_data)
            else:
                print(f"Error: API returned status code {response.status_code}")
                # Fall back to synthetic data if the API fails
                return pd.DataFrame(self._generate_synthetic_data(season, 32))

        except Exception as e:
            print(f"Error fetching current season data: {e}")
            # Fall back to synthetic data if the API fails
            return pd.DataFrame(self._generate_synthetic_data(season, 32))

    def engineer_features(self, data):
        """
        Create features for the model from raw NHL data

        Parameters:
        - data: DataFrame with NHL team data

        Returns:
        - X: Feature matrix
        - y: Target vector (playoff rounds advancement)
        """
        # Create features
        features = data.copy()

        # Calculate additional features
        features['win_percentage'] = features['wins'] / (features['wins'] + features['losses'] + features['ot_losses'])
        features['points_percentage'] = features['points'] / (
                    2 * (features['wins'] + features['losses'] + features['ot_losses']))

        # Calculate team momentum (improvement from previous season)
        # In a real app, you would calculate this properly
        features['momentum'] = np.random.uniform(-0.2, 0.2, size=len(features))

        # One-hot encode categorical variables
        features = pd.get_dummies(features, columns=['team'])

        # Choose features and target
        X_cols = [col for col in features.columns if col not in
                  ['season', 'made_playoffs', 'playoff_rounds', 'won_cup']]

        X = features[X_cols]
        y = features['playoff_rounds']  # Predict how far a team advances

        return X, y

    def train_model(self, X, y):
        """
        Train the prediction model

        Parameters:
        - X: Feature matrix
        - y: Target vector

        Returns:
        - Trained model
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Define model
        model = RandomForestClassifier(random_state=42)

        # Parameter grid for hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }

        # Find best hyperparameters
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train_scaled, y_train)

        # Get best model
        best_model = grid_search.best_estimator_

        # Evaluate
        y_pred = best_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.4f}")

        self.model = best_model
        return best_model

    def predict_playoffs(self, current_season_data):
        """
        Make playoff predictions using the trained model

        Parameters:
        - current_season_data: DataFrame with current season stats

        Returns:
        - DataFrame with playoff predictions
        """
        # Add missing columns that would be in the historical data
        # but aren't in the current season data since playoffs haven't happened yet
        if 'made_playoffs' not in current_season_data.columns:
            current_season_data['made_playoffs'] = 0

        if 'playoff_rounds' not in current_season_data.columns:
            current_season_data['playoff_rounds'] = 0

        if 'won_cup' not in current_season_data.columns:
            current_season_data['won_cup'] = 0

        # Engineer features from current season data
        X_current, _ = self.engineer_features(current_season_data)

        # Ensure X_current has same columns as training data
        missing_cols = set(self.features.columns) - set(X_current.columns)
        for col in missing_cols:
            X_current[col] = 0

        # Ensure columns are in the same order
        X_current = X_current[self.features.columns]

        # Scale features
        X_current_scaled = self.scaler.transform(X_current)

        # Make predictions
        predictions = self.model.predict(X_current_scaled)
        probabilities = self.model.predict_proba(X_current_scaled)

        # Create predictions DataFrame
        results = current_season_data[['team', 'points']].copy()
        results['predicted_playoff_rounds'] = predictions

        # Add probabilities for each outcome
        for i in range(self.model.classes_.max() + 1):
            if i < probabilities.shape[1]:  # Check if the class exists in probabilities
                results[f'prob_round_{i}'] = probabilities[:, i]
            else:
                results[f'prob_round_{i}'] = 0

        # Sort by predicted playoff performance
        results = results.sort_values(by=['predicted_playoff_rounds', 'points'], ascending=False)

        return results

    def simulate_playoff_bracket(self, playoff_teams, num_simulations=1000):
        """
        Simulate the entire playoff bracket using Monte Carlo simulation

        Parameters:
        - playoff_teams: DataFrame with teams that made the playoffs
        - num_simulations: Number of Monte Carlo simulations to run

        Returns:
        - DataFrame with simulation results
        """
        # In a real application, you would:
        # 1. Set up the NHL playoff bracket structure
        # 2. Run simulations of each series
        # 3. Advance winners through the bracket
        # 4. Count final outcomes

        # For this example, we'll create a simplified simulation
        simulation_results = {}

        for team in playoff_teams['team']:
            simulation_results[team] = {
                'conference_finals': 0,
                'finals': 0,
                'champion': 0
            }

        # Simulate playoffs
        for _ in range(num_simulations):
            # This is highly simplified - in reality you'd model the actual bracket
            # and matchups according to NHL playoff rules

            remaining_teams = playoff_teams.copy()

            # Conference finals (final 4 teams)
            conf_finals = remaining_teams.sample(n=4)
            for team in conf_finals['team']:
                simulation_results[team]['conference_finals'] += 1

            # Finals (final 2 teams)
            finals = conf_finals.sample(n=2)
            for team in finals['team']:
                simulation_results[team]['finals'] += 1

            # Champion (1 team)
            champion = finals.sample(n=1)
            champion_team = champion['team'].values[0]
            simulation_results[champion_team]['champion'] += 1

        # Convert to DataFrame
        sim_df = pd.DataFrame(simulation_results).T
        sim_df = sim_df.reset_index().rename(columns={'index': 'team'})

        # Convert to percentages
        for col in ['conference_finals', 'finals', 'champion']:
            sim_df[f'{col}_pct'] = sim_df[col] / num_simulations * 100

        return sim_df.sort_values(by='champion', ascending=False)

    def visualize_bracket(self, simulation_results):
        """
        Create a visualization of the predicted playoff bracket

        Parameters:
        - simulation_results: DataFrame with simulation results

        Returns:
        - None (saves visualization file)
        """
        try:
            # Set the backend to a non-GUI backend to avoid issues
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend

            # Get top teams for visualization
            top_teams = simulation_results.sort_values(by='champion_pct', ascending=False).head(8)

            # Create figure and plot
            plt.figure(figsize=(10, 6))
            bars = sns.barplot(x='champion_pct', y='team', data=top_teams)

            # Add value labels to the bars
            for i, v in enumerate(top_teams['champion_pct']):
                plt.text(v + 0.5, i, f"{v:.1f}%", va='center')

            # Add titles and labels
            plt.title('Stanley Cup Championship Probability (%)')
            plt.xlabel('Probability (%)')
            plt.ylabel('Team')
            plt.tight_layout()

            # Save figure instead of displaying it
            plt.savefig('stanley_cup_predictions.png')
            print(f"Visualization saved to 'stanley_cup_predictions.png'")
            plt.close()

        except Exception as e:
            print(f"Warning: Could not create visualization: {e}")
            print("Continuing without visualization...")

            # Print the top teams instead
            top_teams = simulation_results.sort_values(by='champion_pct', ascending=False).head(8)
            print("\nTop Stanley Cup contenders:")
            for i, row in top_teams.iterrows():
                print(f"{row['team']}: {row['champion_pct']:.1f}%")

    def fetch_player_stats(self, season=2024):
        """
        Fetch player statistics for a given season

        Parameters:
        - season: Season to fetch (start year)

        Returns:
        - DataFrame with player statistics
        """
        season_str = f"{season}{season + 1}"

        # NHL API endpoint for skater stats
        url = f"{self.nhl_api_url}/skater/summary"

        # Parameters for the request
        params = {
            'sort': '[{"property":"points","direction":"DESC"}]',
            'start': 0,
            'limit': 100,  # Get top 100 players
            'factCayenneExp': 'gamesPlayed>=20',  # Only players with at least 20 games
            'cayenneExp': f'seasonId={season_str} and gameTypeId=2'  # Regular season games
        }

        try:
            # Fetch the data
            response = requests.get(url, params=params)

            if response.status_code == 200:
                data = response.json()

                player_data = []
                if 'data' in data:
                    for player in data['data']:
                        player_record = {
                            'player_id': player.get('playerId', 0),
                            'player_name': player.get('skaterFullName', ''),
                            'team': player.get('teamAbbrevs', ''),
                            'position': player.get('positionCode', ''),
                            'games_played': player.get('gamesPlayed', 0),
                            'goals': player.get('goals', 0),
                            'assists': player.get('assists', 0),
                            'points': player.get('points', 0),
                            'plus_minus': player.get('plusMinus', 0),
                            'penalty_minutes': player.get('penaltyMinutes', 0),
                            'time_on_ice_per_game': player.get('timeOnIcePerGame', 0),
                            'shots': player.get('shots', 0),
                            'shooting_pct': player.get('shootingPct', 0),
                        }
                        player_data.append(player_record)

                return pd.DataFrame(player_data)
            else:
                print(f"Error: API returned status code {response.status_code}")
                return pd.DataFrame()  # Return empty DataFrame on error

        except Exception as e:
            print(f"Error fetching player stats: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error

    def fetch_goalie_stats(self, season=2024):
        """
        Fetch goalie statistics for a given season

        Parameters:
        - season: Season to fetch (start year)

        Returns:
        - DataFrame with goalie statistics
        """
        season_str = f"{season}{season + 1}"

        # NHL API endpoint for goalie stats
        url = f"{self.nhl_api_url}/goalie/summary"

        # Parameters for the request
        params = {
            'sort': '[{"property":"wins","direction":"DESC"}]',
            'start': 0,
            'limit': 50,  # Get top 50 goalies
            'factCayenneExp': 'gamesPlayed>=10',  # Only goalies with at least 10 games
            'cayenneExp': f'seasonId={season_str} and gameTypeId=2'  # Regular season games
        }

        try:
            # Fetch the data
            response = requests.get(url, params=params)

            if response.status_code == 200:
                data = response.json()

                goalie_data = []
                if 'data' in data:
                    for goalie in data['data']:
                        goalie_record = {
                            'goalie_id': goalie.get('goalieId', 0),
                            'goalie_name': goalie.get('goalieFullName', ''),
                            'team': goalie.get('teamAbbrevs', ''),
                            'games_played': goalie.get('gamesPlayed', 0),
                            'wins': goalie.get('wins', 0),
                            'losses': goalie.get('losses', 0),
                            'overtime_losses': goalie.get('otLosses', 0),
                            'save_percentage': goalie.get('savePct', 0),
                            'goals_against_avg': goalie.get('goalsAgainstAverage', 0),
                            'shutouts': goalie.get('shutouts', 0),
                        }
                        goalie_data.append(goalie_record)

                return pd.DataFrame(goalie_data)
            else:
                print(f"Error: API returned status code {response.status_code}")
                return pd.DataFrame()  # Return empty DataFrame on error

        except Exception as e:
            print(f"Error fetching goalie stats: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error

    def run_full_prediction(self):
        """Run the entire prediction pipeline"""
        # 1. Fetch historical data
        print("Fetching historical team data...")
        historical_data = self.fetch_historical_data()

        # 2. Engineer features
        print("Engineering features...")
        X, y = self.engineer_features(historical_data)
        self.features = X
        self.target = y

        # 3. Train model
        print("Training model...")
        self.train_model(X, y)

        # 4. Get current season data
        print("Fetching current season team data...")
        current_data = self.fetch_current_season_data()

        # 5. Enhance with player data (optional)
        try:
            print("Fetching player statistics...")
            player_data = self.fetch_player_stats()
            goalie_data = self.fetch_goalie_stats()

            # Here you could integrate player/goalie data with team data
            # This would enhance your model's predictive power
            print(f"Retrieved data for {len(player_data)} players and {len(goalie_data)} goalies")
        except Exception as e:
            print(f"Warning: Could not fetch player data: {e}")
            print("Continuing with team data only...")

        # 6. Make playoff predictions
        print("Making playoff predictions...")
        playoff_predictions = self.predict_playoffs(current_data)

        # 7. Select playoff teams (top 16)
        playoff_teams = playoff_predictions.head(16)

        # 8. Simulate playoff bracket
        print("Simulating playoff bracket...")
        bracket_results = self.simulate_playoff_bracket(playoff_teams)

        # 9. Visualize results
        print("Generating visualization...")
        self.visualize_bracket(bracket_results)

        return {
            'playoff_predictions': playoff_predictions,
            'bracket_results': bracket_results
        }


# Example usage
if __name__ == "__main__":
    # Create predictor
    predictor = StanleyCupPredictor()

    # Get API information
    print("NHL API Information:")
    print(f"Stats API Base URL: {predictor.nhl_api_url}")
    print(f"Web API Base URL: {predictor.nhl_web_api_url}")
    print("\nFetching data from NHL API...")

    # Run prediction pipeline
    results = predictor.run_full_prediction()

    # Display results
    print("\nStanley Cup Predictions for 2025:")
    print(results['bracket_results'][['team', 'champion_pct']].head(5))