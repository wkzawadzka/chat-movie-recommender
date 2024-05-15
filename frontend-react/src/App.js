import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';
import Table from 'react-bootstrap/Table';
import './App.css';
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import movieTitles from './movies.json';
import data from './test1.json';
import { Rings } from 'react-loader-spinner';
import 'bootstrap/dist/css/bootstrap.min.css';
import Card from 'react-bootstrap/Card';
import ListGroup from 'react-bootstrap/ListGroup';
import Modal from 'react-bootstrap/Modal';
import Row from 'react-bootstrap/Row';

function App() {
  const [value, setValue] = useState('');
  const [load, setLoad] = useState('');
  const [empty, setEmpty] = useState(false);
  const [notExist, setNotExist] = useState(false);
  const [movies, setMovies] = useState(false);
  const [showCardContainer, setShowCardContainer] = useState(true); // New state variable
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredMovieTitles, setFilteredMovieTitles] = useState([]);
  const [selectedMovieTitle, setSelectedMovieTitle] = useState('');
  const [showModal, setShowModal] = useState(false); // New state variable
  const [selectedCard, setSelectedCard] = useState(null); // New state variable

  const handleSearchTermChange = (event) => {
    const { value } = event.target;
    if (value.length === 0) {
      setFilteredMovieTitles([]);
      setSearchTerm('');
    } else {
      setSearchTerm(value);

      const filteredTitles = movieTitles.filter((title) =>
        title.toLowerCase().startsWith(value.toLowerCase())
      );
      setFilteredMovieTitles(filteredTitles.slice(0, 5));
    }
  };

  const handleMovieTitleClick = (title) => {
    setSelectedMovieTitle(title);
    setFilteredMovieTitles([]);
    setSearchTerm(title);
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
      const api_key = process.env.REACT_APP_API_KEY;
      const params = { api_key: api_key, query: title };
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

      await axios.get(`http://localhost:8000/get_strategy/` + selectedMovieTitle).then((response) => {
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
        <h2 style={{ margin: '20px', color: 'white' }}>Recommendation movie</h2>
        <form style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', marginBottom: '20px' }}>
          <div>
<input               type="text"               value={searchTerm}               onChange={handleSearchTermChange}               placeholder="Enter a movie title"               style={{ width: '300px' }}             />
            {filteredMovieTitles.length > 0 ? (
              <ListGroup>
                {filteredMovieTitles.map((title, index) => (
                  <ListGroup.Item
                    key={index}
                    onClick={() => handleMovieTitleClick(title)}
                    active={selectedMovieTitle === title}
                    action
                  >
                    {title}
                  </ListGroup.Item>
                ))}
              </ListGroup>
            ) : (
              <p></p>
            )}
            <Button
              variant="primary"
              style={{
                padding: '5px 10px',
                borderRadius: '5px',
                border: 'none',
                backgroundColor: '#4fa94d',
                color: '#fff',
                fontSize: '14px',
                cursor: 'pointer',
              }}
              onClick={fetchData}
            >
              Submit
            </Button>
          </div>
        </form>

        {load && (
<div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '10vh' }}>           <Rings height="150" width="120" color="#4fa94d" radius="6" visible={true} ariaLabel="rings-loading" />         </div>
        )}

{movies && Object.keys(movies).length > 0 && (
      <div className="card-container">
        <h1 style={{ color: 'white' }}>Similar Movies</h1>
        <div className="idex-card-container" style={{ display: 'flex', flexWrap: 'wrap' }}>
          {splitMoviesIntoGroups(movies, 5).map((group, groupIndex) => (
            <div key={groupIndex} style={{ display: 'flex',}}>
              {Object.values(group).map((movie, idx) => {
                return (
                  <div key={idx} style={{ margin: '10px' }}>
                    <ShowMovieDetails
                      movie={movie}
                      onClick={() => handleCardClick(movie)}
                    />
                  </div>
                );
              })}
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
      const api_key = process.env.REACT_APP_API_KEY;
      const params = { api_key: api_key, query: title };
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
