import unittest
from unittest.mock import patch, mock_open
from openvalidators.reward import RewardModelType, RewardFrameworkConfig


class RewardFrameworkConfigTestCase(unittest.TestCase):
    """Test that class is loaded with expected config file, summing to 1"""

    @patch('builtins.open', new_callable=mock_open, read_data=f"""
      reward_models_weights:
          {RewardModelType.rlhf.value}: 0.2
          {RewardModelType.reciprocate.value}: 0.3
          {RewardModelType.dahoas.value}: 0.1
          {RewardModelType.diversity.value}: 0.4
          {RewardModelType.prompt.value}: 0.0
      """)
    def test_load_from_config_file(self, mock_file):
        # Arrange
        config_file_path = 'test_config.yml'
        expected_weights = [0.2, 0.3, 0.1, 0.4, 0.0]

        # Act
        config = RewardFrameworkConfig(config_file_path)

        # Assert
        mock_file.assert_called_once_with(config_file_path, 'r')
        self.assertEqual(config.rlhf_model_weight, expected_weights[0])
        self.assertEqual(config.reciprocate_model_weight, expected_weights[1])
        self.assertEqual(config.dahoas_model_weight, expected_weights[2])
        self.assertEqual(config.diversity_model_weight, expected_weights[3])
        self.assertEqual(config.prompt_model_weight, expected_weights[4])
        self.assertEqual(sum([config.rlhf_model_weight, config.reciprocate_model_weight,
                              config.dahoas_model_weight, config.diversity_model_weight,
                              config.prompt_model_weight]), 1)




    @patch('builtins.open', new_callable=mock_open, read_data=f"""
    # Mock config file
    reward_models_weights:
        {RewardModelType.rlhf.value}: 0.5
        {RewardModelType.reciprocate.value}: 0.5     
    """)
    def test_load_from_config_file_missing_fields_that_sum_to_1(self, mock_file):
        """Test that class is loaded with expected config file with missing fields that sum to 1"""
        # Arrange
        config_file_path = 'test_config.yml'
        expected_weights = [0.5, 0.5]

        # Act
        config = RewardFrameworkConfig(config_file_path)

        # Assert
        mock_file.assert_called_once_with(config_file_path, 'r')

        self.assertEqual(config.rlhf_model_weight, expected_weights[0])
        self.assertEqual(config.reciprocate_model_weight, expected_weights[1])
        self.assertEqual(config.dahoas_model_weight, 0)
        self.assertEqual(config.diversity_model_weight, 0)
        self.assertEqual(config.prompt_model_weight, 0)
        self.assertEqual(sum([config.rlhf_model_weight, config.reciprocate_model_weight,
                              config.dahoas_model_weight, config.diversity_model_weight,
                              config.prompt_model_weight]), 1)

    @patch('builtins.open', new_callable=mock_open, read_data=f"""
    # Mock config file
    reward_models_weights:
        {RewardModelType.rlhf.value}: 0.2
        {RewardModelType.diversity.value}: 0.4
        {RewardModelType.prompt.value}: 0.0
    """)
    def test_load_from_config_file_weights_sum_less_than_1(self, mock_file):
        """Test that class raises Exception when sum of all elements less than one"""
        # Arrange
        config_file_path = 'test_config.yml'

        # Act/Assert
        with self.assertRaises(Exception) as context:
            RewardFrameworkConfig(config_file_path)
        self.assertEqual(str(context.exception), "Sum of reward model weights must be 1")


    @patch('builtins.open', new_callable=mock_open, read_data=f"""
    # Mock config file
    reward_models_weights:
        {RewardModelType.rlhf.value}: 1.0
        {RewardModelType.diversity.value}: 0.1 
        {RewardModelType.prompt.value}: 0.0
    """)
    def test_load_from_config_file_weights_sum_bigger_than_1(self, mock_file):
        """Test that class raises Exception when sum of all elements bigger than one"""
        # Arrange
        config_file_path = 'test_config.yml'

        # Act/Assert
        with self.assertRaises(Exception) as context:
            RewardFrameworkConfig(config_file_path)
        self.assertEqual(str(context.exception), "Sum of reward model weights must be 1")

