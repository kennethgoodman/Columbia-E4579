import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { getRefreshTokenIfExists } from '../../utils/tokenHandler';
import './Poll.css'; 
import { useContext } from 'react';
import { DarkModeContext } from '../../components/darkmode/DarkModeContext';

const AdminPoll = () => {
  const [polls, setPolls] = useState([]);
  const [newPollData, setNewPollData] = useState({
    question: '',
    available: false,
    choices: [{ text: '' }],
  });
  const [errorMessage, setErrorMessage] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null); 
  const navigate = useNavigate();
  const { darkMode } = useContext(DarkModeContext);

  useEffect(() => {
    const fetchPolls = async () => {
      try {
        const response = await axios.get(`${process.env.REACT_APP_API_SERVICE_URL}/polls`, {
          headers: {
            Authorization: `Bearer ${getRefreshTokenIfExists()}`
          }
        });
        setPolls(response.data);
      } catch (error) {
        // Handle errors, including 401 Unauthorized
        if (error.response && error.response.status === 401) {
          navigate('/login');
        } else {
          setErrorMessage('Error loading polls. Please try again later.');
          console.error('Error fetching polls:', error);
        }
      }
    };

    fetchPolls();
  }, [navigate]);

  const handleInputChange = (event) => {
    const { name, value, type, checked } = event.target;
    const newValue = type === 'checkbox' ? checked : value;
    setNewPollData({ ...newPollData, [name]: newValue });
  };

  const handleChoiceChange = (index, event) => {
    const updatedChoices = [...newPollData.choices];
    updatedChoices[index].text = event.target.value; 
    setNewPollData({ ...newPollData, choices: updatedChoices });
  };

  const addChoiceField = () => {
    setNewPollData({
      ...newPollData,
      choices: [...newPollData.choices, { text: '' }],
    });
  };

  const removeChoiceField = (index) => {
    const updatedChoices = [...newPollData.choices];
    updatedChoices.splice(index, 1);
    setNewPollData({ ...newPollData, choices: updatedChoices });
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      const response = await axios.post(
        `${process.env.REACT_APP_API_SERVICE_URL}/polls`,
        newPollData,
        {
          headers: {
            Authorization: `Bearer ${getRefreshTokenIfExists()}`,
            'Content-Type': 'application/json',
          },
        }
      );

      if (response.status === 201) {
        setPolls([...polls, response.data]); 
        setNewPollData({ question: '', available: false, choices: [{ text: '' }] }); 
        setSuccessMessage('Poll created successfully!');
      } else {
        setErrorMessage('Error creating poll. Please try again.'); 
      }
    } catch (error) {
      setErrorMessage('Error creating poll. Please try again.'); 
      console.error('Error creating poll:', error);
    }
  };

  const togglePollAvailability = async (pollId, currentAvailability) => {
    try {
      await axios.put( 
        `${process.env.REACT_APP_API_SERVICE_URL}/polls/${pollId}`,
        { available: !currentAvailability },
        {
          headers: {
            Authorization: `Bearer ${getRefreshTokenIfExists()}`,
            'Content-Type': 'application/json',
          },
        }
      );

      setPolls(polls.map((poll) =>
        poll.id === pollId ? { ...poll, available: !currentAvailability } : poll
      ));
      setSuccessMessage('Poll availability updated!'); 
    } catch (error) {
      setErrorMessage('Error updating poll availability. Please try again.'); 
      console.error(`Error toggling poll availability for poll ID ${pollId}:`, error);
    }
  };

  const handleDeletePoll = async (pollId) => {
    try {
      await axios.delete(
        `${process.env.REACT_APP_API_SERVICE_URL}/polls/${pollId}`,
        {
          headers: {
            Authorization: `Bearer ${getRefreshTokenIfExists()}`,
          },
        }
      );

      setPolls(polls.filter((poll) => poll.id !== pollId));
      setSuccessMessage('Poll deleted successfully!'); 
    } catch (error) {
      setErrorMessage('Error deleting poll. Please try again.');
      console.error(`Error deleting poll ID ${pollId}:`, error);
    }
  };

  return (
    <div className={`poll-container ${darkMode ? 'dark' : ''}`}> 
      <h2>Admin Panel</h2>

      {errorMessage && <div className="error-message">{errorMessage}</div>}
      {successMessage && (
        <div className="success-message">{successMessage}</div>
      )}
      <h3>Create New Poll</h3>
      <form onSubmit={handleSubmit}>
        <div>
          <label htmlFor="question">Question:</label>
          <input
            type="text"
            id="question"
            name="question"
            value={newPollData.question}
            onChange={handleInputChange}
          />
        </div>
        <div>
          <label htmlFor="available">Available:</label>
          <input
            type="checkbox"
            id="available"
            name="available"
            checked={newPollData.available}
            onChange={handleInputChange}
          />
        </div>
        <h4>Choices</h4>
        {newPollData.choices.map((choice, index) => (
          <div key={index}>
            <input
              type="text"
              value={choice.text}
              onChange={(event) => handleChoiceChange(index, event)}
              placeholder={`Choice ${index + 1}`}
            />
            <button type="button" onClick={() => removeChoiceField(index)}>
              Remove
            </button>
          </div>
        ))}
        <button type="button" onClick={addChoiceField}>
          Add Choice
        </button>
        <button type="submit">Create Poll</button>
      </form>

      <h3>Manage Polls</h3>
      <ul>
        {polls.map((poll) => (
          <li key={poll.id}>
            <h4>{poll.question}</h4>
            <p>Available: {poll.available ? 'Yes' : 'No'}</p>
            <button onClick={() => togglePollAvailability(poll.id, poll.available)}>
              {poll.available ? 'Lock' : 'Unlock'}
            </button>
            <button onClick={() => handleDeletePoll(poll.id)}>
              Delete
            </button>
            <ul>
              {poll.choices.map((choice) => (
                <li key={choice.id}>{choice.text}</li>
              ))}
            </ul>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default AdminPoll;