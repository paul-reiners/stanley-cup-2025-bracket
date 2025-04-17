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


class StanleyCupPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.teams_data = {}
        self.features = None
        self.target = None

    def fetch_historical_data(self, start_season=2010, end_season=2024):
        """
        Fetch historical NHL data from the NHL API or a similar source

        Parameters:
        - start_season: First season to include (start year)
        - end_season: Last season to include (end year)

        Returns:
        - DataFrame with historical team and playoff data
        """
        all_data = []

        for season in range(start_season, end_season + 1):
            season_str = f"{season}{season + 1}"

            # This is a placeholder - you'd need to use the actual NHL API endpoints
            # Example URL for NHL API (update with correct endpoints)
            url = f"https://statsapi.web.nhl.com/api/v1/teams?season={season_str}"

            try:
                # In a real application, you would fetch the actual data
                # response = requests.get(url)
                # data = response.json()

                # For demonstration, we'll create synthetic data
                data = self._generate_synthetic_data(season, 32)  # 32 NHL teams
                all_data.extend(data)

            except Exception as e:
                print(f"Error fetching data for season {season_str}: {e}")

        # Convert to DataFrame
        df = pd.DataFrame(all_data)
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
        """Fetch data for the current season to make predictions"""
        # Similar to fetch_historical_data but only for current season
        # In a real app, you would fetch real-time data
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
        # Engineer features from current season data
        X_current, _ = self.engineer_features(current_season_data)

        # Ensure X_current has same columns as training data
        missing_cols = set(self.features.columns) - set(X_current.columns)
        for col in missing_cols:
            X_current[col] = 0

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
            results[f'prob_round_{i}'] = probabilities[:, i] if i < len(self.model.classes_) else 0

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
        """Create a visualization of the predicted playoff bracket"""
        # This would create a visual representation of the predicted bracket
        # For simplicity, we'll just show the championship probabilities

        top_teams = simulation_results.sort_values(by='champion_pct', ascending=False).head(8)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='champion_pct', y='team', data=top_teams)
        plt.title('Stanley Cup Championship Probability (%)')
        plt.xlabel('Probability (%)')
        plt.ylabel('Team')
        plt.tight_layout()

        # In a real application, you would save this figure or display it
        # plt.savefig('stanley_cup_predictions.png')

    def run_full_prediction(self):
        """Run the entire prediction pipeline"""
        # 1. Fetch historical data
        historical_data = self.fetch_historical_data()

        # 2. Engineer features
        X, y = self.engineer_features(historical_data)
        self.features = X
        self.target = y

        # 3. Train model
        self.train_model(X, y)

        # 4. Get current season data
        current_data = self.fetch_current_season_data()

        # 5. Make playoff predictions
        playoff_predictions = self.predict_playoffs(current_data)

        # 6. Select playoff teams (top 16)
        playoff_teams = playoff_predictions.head(16)

        # 7. Simulate playoff bracket
        bracket_results = self.simulate_playoff_bracket(playoff_teams)

        # 8. Visualize results
        self.visualize_bracket(bracket_results)

        return {
            'playoff_predictions': playoff_predictions,
            'bracket_results': bracket_results
        }


# Example usage
if __name__ == "__main__":
    # Create predictor
    predictor = StanleyCupPredictor()

    # Run prediction pipeline
    results = predictor.run_full_prediction()

    # Display results
    print("\nStanley Cup Predictions for 2025:")
    print(results['bracket_results'][['team', 'champion_pct']].head(5))
