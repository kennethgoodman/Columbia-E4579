import React, { useEffect, useRef, useState, useContext } from "react";
import LikeButton from "../Likes/LikeButton";
import DislikeButton from "../Likes/DislikeButton";
import axios from "axios";
import { getRefreshTokenIfExists } from "../../utils/tokenHandler";
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { useSpring, animated } from 'react-spring';
import { DarkModeContext } from '../../components/darkmode/DarkModeContext';

import "./Post.css";

const get_options = (uri, content_id) => {
  let api_uri = `${process.env.REACT_APP_API_SERVICE_URL}/engagement/${uri}/${content_id}`;
  return {
    url: api_uri,
    method: "post",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${getRefreshTokenIfExists()}`,
    }, 
  };
};

const Post = (props) => {
  const { darkMode } = useContext(DarkModeContext);
  const [likeIsClicked, setLikeIsClicked] = useState(props.post.user_likes);
  const [dislikeIsClicked, setDislikeIsClicked] = useState(props.post.user_dislikes);
  const [totalLikes, setTotalLikes] = useState(props.post.total_likes);
  const [totalDislikes, setTotalDislikes] = useState(props.post.total_dislikes);
  const [isAuthenticated, _] = useState(getRefreshTokenIfExists() !== null);

  const image_ref = useRef(null);
  useIntersectionObserver(props.controller, props.content_id, image_ref, {});

  const like = (callback) => {
    axios(get_options("like", props.content_id))
      .then(callback)
      .catch(function (error) {
        toast.error('Something went wrong, please try again.');
      });
  };
  const unlike = (callback) => {
    axios(get_options("unlike", props.content_id))
      .then(callback)
      .catch(function (error) {
        toast.error('Something went wrong, please try again.');
      });
  };
  const dislike = (callback) => {
    axios(get_options("dislike", props.content_id))
      .then(callback)
      .catch(function (error) {
        toast.error('Something went wrong, please try again.');
      });
  };
  const undislike = (callback) => {
    axios(get_options("undislike", props.content_id))
      .then(callback)
      .catch(function (error) {
        toast.error('Something went wrong, please try again.');
      });
  };

  const handleLikes = () => {
    if (likeIsClicked) {
      unlike((_) => {
        setLikeIsClicked(false); // unclick it
        setTotalLikes(totalLikes - 1);
      });
    } else {
      like((_) => {
        setLikeIsClicked(true); // click it
        setTotalLikes(totalLikes + 1);
      });
      if (dislikeIsClicked) {
        setDislikeIsClicked(false);
        setTotalDislikes(totalDislikes - 1);
      }
    }
  };

  const handleDislikes = () => {
    console.log(
      `handing dislike for ${props.content_id}, ${likeIsClicked}, ${dislikeIsClicked}`,
    );
    if (dislikeIsClicked) {
      undislike((_) => {
        setDislikeIsClicked(false); // unclick it
        setTotalDislikes(totalDislikes - 1);
      });
    } else {
      dislike((_) => {
        setDislikeIsClicked(true); // click it
        setTotalDislikes(totalDislikes + 1);
      });
      if (likeIsClicked) {
        setLikeIsClicked(false);
        setTotalLikes(totalLikes - 1);
      }
    }
  };

  const likeButtonSpring = useSpring({
    opacity: 1,
    from: { opacity: 0 },
  });
  
  const dislikeButtonSpring = useSpring({
    opacity: 1,
    from: { opacity: 0 },
  });

  const renderPostContent = () => {
    switch (props.post.type) {
      case 'image':
        return (
          <img
            ref={contentRef}
            src={props.post.download_url}
            alt={props.post.text}
            onDoubleClick={handleLikes}
          />
        );
      case 'text':
        return (
          <div className={`textPostContent ${darkMode ? 'dark' : ''}`} ref={contentRef}>
            {props.post.text}
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className={`postContainer ${darkMode ? 'dark' : ''}`}>
      <ToastContainer />
      <h4 className={`postAuthor ${darkMode ? 'dark' : ''}`}>{props.post.author}</h4>
      {renderPostContent()}
      {isAuthenticated && (
        <div className={`likesContainer ${darkMode ? 'dark' : ''}`}>
          <animated.div style={likeButtonSpring}>
            <LikeButton
              content_id={props.content_id}
              total_likes={totalLikes}
              user_likes={likeIsClicked}
              handleLikes={handleLikes}
            />
          </animated.div>
          {totalLikes - totalDislikes}
          <animated.div style={dislikeButtonSpring}>
            <DislikeButton
              content_id={props.content_id}
              total_dislikes={totalDislikes}
              user_dislikes={dislikeIsClicked}
              handleDislikes={handleDislikes}
            />
          </animated.div>
        </div>
      )}
    </div>
  );
};

function useIntersectionObserver(
  controller,
  content_id,
  elementRef,
  {
    threshold = 0.9,
    root = null,
    rootMargin = "0%",
    freezeOnceVisible = false,
  },
) {
  const [entry, setEntry] = useState();
  const [startIntersecting, setStartIntersecting] = useState(-1);

  const frozen = entry?.isIntersecting && freezeOnceVisible;

  const handle_elapsed = (elapsed_time, content_id) => {
    if (elapsed_time <= 250) {
      return;
    }
    console.log("in handle elapsed", elapsed_time, content_id);
    let api_uri = `${process.env.REACT_APP_API_SERVICE_URL}/engagement/elapsed_time`;
    axios({
      ...get_options("elapsed_time", content_id),
      data: JSON.stringify({ elapsed_time: elapsed_time, controller: controller }),
    }).catch(function (error) {
      console.log(error);
    });
  };

  const updateEntry = ([entry]) => {
    setEntry(entry);
    if (entry.isIntersecting) {
      if (startIntersecting !== -1) {
        return;
      }
      setStartIntersecting(Date.now());
      // handleLoad();
    } else if (startIntersecting === -1) {
      return;
      // first time loading, but not in view
    } else {
      // not intersecting, and not loaded for first time
      handle_elapsed(Date.now() - startIntersecting, content_id);
      setStartIntersecting(-1);
    }
  };

  useEffect(() => {
    // https://usehooks-ts.com/react-hook/use-intersection-observer
    const node = elementRef?.current; // DOM Ref
    const hasIOSupport = !!window.IntersectionObserver;

    if (!hasIOSupport || frozen || !node) return;

    const observerParams = { threshold, root, rootMargin };
    const observer = new IntersectionObserver(updateEntry, observerParams);

    observer.observe(node);

    return () => observer.disconnect();
  }, [
    elementRef,
    JSON.stringify(threshold),
    root,
    rootMargin,
    frozen,
    startIntersecting,
    content_id,
  ]);

  return entry;
}

export default Post;
