import Button from 'react-bootstrap/Button';
import './App.css';
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Rings } from 'react-loader-spinner';
import 'bootstrap/dist/css/bootstrap.min.css';
import Card from 'react-bootstrap/Card';
import ListGroup from 'react-bootstrap/ListGroup';
import Modal from 'react-bootstrap/Modal';
import { parse } from 'ini';

const API_KEY = "55faacca1ec79003b9a8b9e4c3819c99";

function App() {
  const [value, setValue] = useState('');
  const [load, setLoad] = useState('');
  const [empty, setEmpty] = useState(false);
  const [notExist, setNotExist] = useState(false);
  const [movies, setMovies] = useState(false);
  const [showCardContainer, setShowCardContainer] = useState(true); // New state variable
  const [searchQuery, setSearchQuery] = useState('');
  const [filteredMovieTitles, setFilteredMovieTitles] = useState([]);
  const [selectedMovieTitle, setSelectedMovieTitle] = useState('');
  const [showModal, setShowModal] = useState(false); // New state variable
  const [selectedCard, setSelectedCard] = useState(null); // New state variable
  const movieTitles = ["Toy Story (1995)"];
  const [isEditing, setIsEditing] = useState(false);


  const handleSearchQueryChange = (event) => {
    setSearchQuery(event.target.value);
  };

  const handleKeyDown = async (event) => {
    if (event.key === 'Enter') {
      setIsEditing(false);
      event.preventDefault(); // Prevent default form submission behavior
      await fetchData();
      event.target.blur(); // Remove focus from the input field
      setSelectedMovieTitle("xd");
    }
  };

  const handleClick = () => {
    setIsEditing(true);
  };

  const handleMovieTitleClick = (title) => {
    setSelectedMovieTitle(title);
    setFilteredMovieTitles([]);
    setSearchQuery(title);
  };
  const handleCardClick = (card) => {
    setSelectedCard(card);
    fetchMovieImage(card);
    setShowModal(true);
  };

  const fetchMovieImage = async (card) => {
    const v = 'title';
    if (card.hasOwnProperty(v)) {
      let title = card[v];
      title = title.split('(')[0].trim();
      const base_url = 'https://api.themoviedb.org/3/search/movie';
      const params = { api_key: API_KEY, query: title };
      const response = await axios.get(base_url, { params });
      if (typeof response.data.results[0].poster_path !== 'undefined') {
        const poster_path = response.data.results[0].poster_path;
        const base_img_url = 'https://image.tmdb.org/t/p/';
        const size = 'w500';
        const img_url = `${base_img_url}${size}${poster_path}`;
        setSelectedCard((prevCard) => ({
          ...prevCard,
          imgUrl: img_url
        }));
      }
    }
  };

  const splitMoviesIntoGroups = (movies, groupSize) => {
    const movieKeys = Object.keys(movies[0]);
    const groups = [];
    let currentGroup = {};
    let i = 0;
    movieKeys.forEach((key, index) => {
      i = i + 1;
      const movie = movies[0][key];
      console.log(movie)
      currentGroup[key] = movie;

      if ((index + 1) % groupSize === 0 || index === movieKeys.length - 1) {
        groups.push(currentGroup);
        currentGroup = {};
      }
    });
    return groups;
  };

  const toggleCardContainer = () => {
    setShowCardContainer(!showCardContainer);
  };

  const handleCloseModal = () => {
    setShowModal(false);
  };

  async function fetchData() {
    setMovies(false);
    setEmpty(false);
    setNotExist(false);

    if (selectedMovieTitle === '') {
      setEmpty(true);
    } else {
      setLoad(true);
      await axios.get(`http://localhost:8000/recommend/` + searchQuery).then((response) => {
        if (response.data[0] !== false) {
          setMovies(response.data);
          setLoad(false);
        } else {
          setLoad(false);
          setNotExist(true);
        }
      });
    }
  }

  function handleInputChange(event) {
    setValue(event.target.value);
  }

  useEffect(() => {
    setLoad(false);
    setEmpty(false);
    setMovies(false);
    setNotExist(false);
  }, []);

  return (
    <div className="App">
      <div>
        <h2 style={{ marginTop: '80px', marginBottom:'20px', color: 'white' }}>What kind of movie are you looking for?</h2>
        <form style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', marginBottom: '20px'}}>
        <div onClick={handleClick} style={{ display: 'flex', flexWrap: 'wrap' }}>
          <textarea
            value={searchQuery}
            style={{ width: '500px', height: '100px', overflowWrap: 'break-word', resize: 'none' }}
            onChange={handleSearchQueryChange}
            onKeyDown={handleKeyDown}
            placeholder="An animation movie about toys come alive"
          />
        </div>
      </form>

        {load && (
<div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '10vh' }}>           <Rings height="150" width="120" color="#4fa94d" radius="6" visible={true} ariaLabel="rings-loading" />         </div>
        )}

{movies && Object.keys(movies).length > 0 && (
  <div className="card-container">
    <h2 style={{ color: 'white' }}>We recommend you</h2>
    <div className="idex-card-container" style={{ display: 'flex', flexWrap: 'wrap' }}>
      {splitMoviesIntoGroups(movies, 5).map((group, groupIndex) => (
        <div key={groupIndex} style={{ display: 'flex',}}>
          {groupIndex === 3 && (
            <div style={{ marginBottom: '10px' }}>
              <span style={{ color: 'green' }}>Bert</span>
            </div>
          )}

          {groupIndex === 0 && (
            <div style={{ marginBottom: '10px' }}>
              <span style={{ color: 'blue' }}>T5</span>
            </div>
          )}

          {groupIndex === 1 && (
              <div style={{ marginBottom: '10px' }}>
                <span style={{ color: 'red' }}>TF-IDF</span>
              </div>
            )}

          {groupIndex ===  2 && (
              <div style={{ marginBottom: '10px' }}>
                <span style={{ color: 'pink' }}>Word2Vec</span>
              </div>
            )}

          {/* Render movies in the group */}
          {Object.values(group).map((movie, idx) => (
            <div key={idx} style={{ margin: '10px' }}>
              <ShowMovieDetails
                movie={movie}
                onClick={() => handleCardClick(movie)}
              />
            </div>
          ))}
        </div>
      ))}
    </div>
  </div>
)}


        <Modal show={showModal} onHide={handleCloseModal}>
          <Modal.Header closeButton>
            <Modal.Title>{selectedCard?.title}</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <Card.Img variant="top" src={selectedCard?.imgUrl} />
            <p>{selectedCard?.overview}</p>
            {selectedCard?.genre && (
              <p>
                <strong>Genre:</strong> {selectedCard.genre.join(', ')}
              </p>
            )}
            {selectedCard?.actors && (
              <p>
                <strong>Actors:</strong> {selectedCard.actors.join(', ')}
              </p>
            )}
            {/* Additional card information can be displayed here */}
          </Modal.Body>
          <Modal.Footer>
            <Button variant="secondary" onClick={handleCloseModal}>
              Close
            </Button>
          </Modal.Footer>
        </Modal>
      </div>
    </div>
  );
}

function ShowMovieDetails(props) {
  const { title, overview, genre, actors } = props.movie;
  const [imgUrl, setImgUrl] = useState('');

  useEffect(() => {
    fetchMovieImage();
  }, []);

  const fetchMovieImage = async () => {
    const v = 'title';
    if (props.movie.hasOwnProperty(v)) {
      let title = props.movie[v];
      title = title.split('(')[0].trim();
      const base_url = 'https://api.themoviedb.org/3/search/movie';
      const params = { api_key: API_KEY, query: title };
      const response = await axios.get(base_url, { params });
      if (typeof response.data.results[0].poster_path !== 'undefined') {
        const poster_path = response.data.results[0].poster_path;
        const base_img_url = 'https://image.tmdb.org/t/p/';
        const size = 'w500';
        const img_url = `${base_img_url}${size}${poster_path}`;
        setImgUrl(img_url);
      }
    }
  };

  return (
    <Card style={{ width: '18rem', height: '100%', display: 'flex' }} onClick={props.onClick}>
      <Card.Img variant="top" src={imgUrl} />
      <Card.Body>
        <Card.Title>{title}</Card.Title>
        <Card.Text>{overview.slice(0, 100)}</Card.Text>
      </Card.Body>
    </Card>
  );
}

export default App;
