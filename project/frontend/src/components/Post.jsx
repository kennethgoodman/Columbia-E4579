const Post = ({ post }) => {
  return (
    <div className="post">
      <h4>{post.author}</h4>
      <img src={post.download_url} alt={post.author} />
    </div>
  );
};

export default Post;
