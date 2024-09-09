import React, { useContext } from "react";
import { DarkModeContext } from '../../components/darkmode/DarkModeContext';

const About = () => {
  const { darkMode } = useContext(DarkModeContext);

  const styles = {
    container: {
      fontFamily: 'Arial, sans-serif',
      margin: '2rem',
    },
    title: {
      borderBottom: '2px solid #ddd',
      paddingBottom: '0.5em',
    },
    content: {
      fontSize: '1rem',
      lineHeight: '1.6',
      marginBottom: '1.5em',
    },
    subTitle: {
      fontWeight: 'bold',
      fontSize: '1.2rem',
      marginTop: '1em',
    },
    list: {
      marginLeft: '2rem',
      listStyle: 'disc',
    },
    table: {
      borderCollapse: 'collapse',
      width: '100%',
      marginBottom: '1rem',
    },
    th: {
      backgroundColor: '#f0f0f0',
      border: '1px solid #ddd',
      padding: '0.5rem',
    },
    td: {
      border: '1px solid #ddd',
      padding: '0.5rem',
    },
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Columbia E4579: Modern Recommendation Systems (Fall 2024)</h1>

      <p style={styles.content}>Welcome to IEOR 4579! This course will guide you through building an end-to-end recommendation system. Feel free to ask questions and discuss the material - the more you participate, the more you'll learn.</p>

      <h2>Course Description</h2>

      <p>In this course, you'll leverage student engagement data to create a recommendation app. This app will utilize AI-generated photos and text and require you to recommend a feed from over 500,000 pieces of AI generated content.</p>

      <p>These concepts are applicable to various recommendation systems, from e-commerce to travel to social media to financial modeling. The instructor's experience at Uber Eats, Facebook, Instagram, and Google will provide valuable insights into real-world use cases.</p>

      <h2>Schedule</h2>

      <table>
        <thead>
          <tr>
            <th>Date</th>
            <th>Topic</th>
            <th>Due</th>
            <th>Reading</th>
            <th>In Class Participation</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>9/9/2024</td>
            <td>Introduction To Yourselves, Myself, The Course and Recommendation Systems</td>
            <td>Survey (1 point - 1 cumulative points)</td>
            <td>N/A</td>
            <td>(1 point - 2 cumulative points)</td>
          </tr>
          <tr>
            <td>9/16/2024</td>
            <td>Implicit and Explicit Embeddings, Taxonomies And Data Privacy</td>
            <td>Engagement Part 0 (10 points - 12 cumulative points)</td>
            <td><a href="https://jalammar.github.io/illustrated-word2vec/" target="_blank" rel="noreferrer noopener">Illustrated Word2vec</a></td>
            <td>(1 point - 13 cumulative points)</td>
          </tr>
          <tr>
            <td>9/23/2024</td>
            <td>Candidate Generation</td>
            <td>Taxonomy Or Embedding Creation (10 points - 23 cumulative points)</td>
            <td><a href="https://arxiv.org/abs/2007.07203" target="_blank" rel="noreferrer noopener">Pre-training Tasks for Embedding-based Large-scale Retrieval</a></td>
            <td>(1 point - 24 cumulative points)</td>
          </tr>
          <tr>
            <td>9/30/2024</td>
            <td>Trending & AI Ethics</td>
            <td>Candidate Generator (5 points - 29 cumulative points)</td>
            <td><a href="https://en.wikipedia.org/wiki/Standard_score" target="_blank" rel="noreferrer noopener">Standard Score (Wikipedia)</a></td>
            <td>(3 points - 32 cumulative points)</td>
          </tr>
          <tr>
            <td>10/7/2024</td>
            <td>LLMs Part 1 (Transformer Architecture, Attention, Model Distillation)</td>
            <td>Teach an LLM part 1 (5 points - 37 cumulative points)</td>
            <td><a href="https://towardsdatascience.com/illustrated-guide-to-transformers-step-by-step-explanation-f74876522bc0" target="_blank" rel="noreferrer noopener">Illustrated Guide to Transformers</a></td>
            <td>(1 point - 38 cumulative points)</td>
          </tr>
          <tr>
            <td>10/14/2024</td>
            <td>Infrastructure</td>
            <td>Build An LLM (10 points - 48 cumulative points)</td>
            <td>
              <ul>
                <li><a href="https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/" target="_blank" rel="noreferrer noopener">Faiss: A Library for Efficient Similarity Search</a></li>
                <li><a href="https://nlp.stanford.edu/IR-book/html/htmledition/a-first-take-at-building-an-inverted-index-1.html" target="_blank" rel="noreferrer noopener">A First Take at Building an Inverted Index</a></li>
              </ul>
            </td>
            <td>(1 point - 49 cumulative points)</td>
          </tr>
          <tr>
            <td>10/21/2024</td>
            <td>LLMs Part 2</td>
            <td>Engagement Part 1 (5 points - 54 cumulative points)</td>
            <td><a href="https://arxiv.org/abs/2106.09685" target="_blank" rel="noreferrer noopener">Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity</a></td>
            <td>(1 point - 55 cumulative points)</td>
          </tr>
          <tr>
            <td>10/28/2024</td>
            <td>Misc Topics (Explore/Exploit & Cold Start + Caching + Filtering + Federated Learning)</td>
            <td>Build an LLM Part 2 (10 points - 65 cumulative points)</td>
            <td><a href="https://arxiv.org/abs/1904.05046" target="_blank" rel="noreferrer noopener">Recommending What Video to Watch Next: A Multimodal Content-aware Approach</a></td>
            <td>(1 point - 66 cumulative points)</td>
          </tr>
          <tr>
            <td>11/4/2024</td>
            <td>No Class</td>
            <td>Explore/Exploit & Cold Start (10 points - 76 cumulative points)</td>
            <td>N/A</td>
            <td>N/A</td>
          </tr>
          <tr>
            <td>11/11/2024</td>
            <td>Real World Use Cases + Anonymity & Data Sharing</td>
            <td>Engagement Part 2 (10 points - 86 cumulative points)</td>
            <td><a href="https://ai.meta.com/blog/ai-unconnected-content-recommendations-facebook-instagram/" target="_blank" rel="noreferrer noopener">AI-powered Content Recommendations at Facebook and Instagram</a></td>
            <td>(1 point - 87 cumulative points)</td>
          </tr>
          <tr>
            <td>11/18/2024</td>
            <td>Prediction + Ranking</td>
            <td>Paper on privacy (10 points - 97 cumulative points)</td>
            <td><a href="https://arxiv.org/pdf/1904.06813" target="_blank" rel="noreferrer noopener">Deep Learning Recommendation Model for Personalization and Recommendation Systems</a></td>
            <td>(1 point - 98 cumulative points)</td>
          </tr>
          <tr>
            <td>11/25/2024</td>
            <td>Ground Truth Data + Temporal/Seasonality</td>
            <td>Final Stage Ranking (10 points - 108 cumulative points)</td>
            <td><a href="http://www.ijicic.org/ijicic-130511.pdf" target="_blank" rel="noreferrer noopener">Building Context-Aware Recommender System Using Hybrid Approach</a></td>
            <td>(1 point - 109 cumulative points)</td>
          </tr>
          <tr>
            <td>12/2/2024</td>
            <td>MMOE + Out Of Sample Distributions</td>
            <td>N/A (Thanksgiving break)</td>
            <td><a href="https://machinelearningmastery.com/mixture-of-experts/" target="_blank" rel="noreferrer noopener">Mixture of Experts for Machine Learning</a></td>
            <td>(1 point - 110 cumulative points)</td>
          </tr>
          <tr>
            <td>12/9/2024</td>
            <td>Wrap Up + Final Project</td>
            <td>N/A (Start working on your last project!)</td>
            <td>N/A</td>
            <td>N/A</td>
          </tr>
          <tr>
            <td>12/16/2024</td>
            <td>N/A</td>
            <td>Teach an LLM part 2 (10 points - 120 cumulative points)</td>
            <td>N/A</td>
            <td>N/A</td>
          </tr>
        </tbody>
      </table>

      <h2>Grading</h2>

      <p>As you can see, there are more than 100 points in the class, there will be no curving. Your grade will be calculated as Min(grade, 100%) and then mapped to Columbia's guidelines:</p>

      <ul>
        <li>A+ = 98-100%</li>
        <li>A  = 93-97.9%</li>
        <li>A- = 90-92.9%</li>
        <li>B+ = 87-89.9%</li>
        <li>B  = 83-86.9%</li>
        <li>B- = 80-82.9%</li>
        <li>C+ = 77-79.9%</li>
        <li>C  = 73-76.9%</li>
        <li>C- = 70-72.9%</li>
        <li>D = 60-69.9%</li>
        <li>F = less than 59.9% </li>
      </ul>

      <p>There will be no rounding. Any late assignment will not be accepted and is worth 0 points. You have 20 points of leeway if something personal comes up.</p>

      <h2>Coding Assignments</h2>

      <p>Coding assignments will be done in Google Colab and submitted in Canvas and can be completed on the free accounts. You are welcome, not required, to use higher powered GPUs if that interests you, but should provide zero advantage.</p>

      <h2>Miscellaneous</h2>

      <p>I reserve the right to change the assignments at any time for any reason. The general structure of the class should not change materially.</p>

      <p>In class participation will be done online. Please bring a cellphone or computer with access to the internet (or wifi), if this is not possible, please let me know as soon as possible.</p>

    </div>
  )
};

export default About;
