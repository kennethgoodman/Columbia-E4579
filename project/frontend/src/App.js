import InfiniteScroll from "react-infinite-scroller";
import { useInfiniteQuery } from "react-query";

import Post from "./components/Post";
import "./App.css";

export default function App() {
  const fetchPosts = async ({ pageParam = 1 }) => {
    const response = await fetch(
      `https://picsum.photos/v2/list?page=${pageParam}&limit=10`
    );
    const results = await response.json();
    return { results, nextPage: pageParam + 1, totalPages: 100 };
  };

  const {
    data,
    isLoading,
    isError,
    hasNextPage,
    fetchNextPage
  } = useInfiniteQuery("posts", fetchPosts, {
    getNextPageParam: (lastPage, pages) => {
      if (lastPage.nextPage < lastPage.totalPages) return lastPage.nextPage;
      return undefined;
    }
  });

  return (
    <div className="App">
      <main>
        {isLoading ? (
          <p>Loading...</p>
        ) : isError ? (
          <p>There was an error</p>
        ) : (
          <InfiniteScroll hasMore={hasNextPage} loadMore={fetchNextPage}>
            {data.pages.map((page) =>
              page.results.map((post) => <Post key={post.id} post={post} />)
            )}
          </InfiniteScroll>
        )}
      </main>
    </div>
  );
}
