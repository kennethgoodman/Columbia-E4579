import LikeButton from "./Like";

const Post = ({ content_id, post }) => {
  return (
    <div className="post">
      <h4>{post.author}</h4>
      <img src={post.download_url} alt={post.text} />
        <LikeButton content_id={content_id}></LikeButton>
    </div>
  );
};

export default Post;
