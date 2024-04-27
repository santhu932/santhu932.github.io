window.projects = [

  {
    Title: "Scaling Inference with Multi-GPU Architectures",
    Description:
      "<ul class='project-ul'><li>Implemented a simple MLP neural network for uncertainty estimation via Monte Carlo Droput, utilizing dropout layers to sample from the predictive distribution of the model.</li>\
      <li>Parallel computation of multiple samples on multi-GPU using Pytorch multiprocessing and DDP libraries.</li></ul>",
    Year: "2024",
    Date: "Feb 21, 2024",
    Skills: ["Python", "Pytorch", "Multiprocessing", "Distributed Data Parallel", "Uncertainty Estimation", "MC Dropout", "Ensembles",],
    TeamSize: 1,
    GithubLink: "https://medium.com/@santhoshpatil932/scaling-inference-with-multi-gpu-architectures-a-deep-dive-into-uncertainty-estimation-0008ae7a106a",
  },

  {
    Title: "NLP-Driven Market Sentiment & Valuation Analysis",
    Description:
      "<ul class='project-ul'><li>Developed a sophisticated NLP pipeline with Amazon SageMaker, Hugging Face, Pegasus for summarization, and FinBERT for sentiment analysis, enhancing market sentiment and company valuation insights from SEC filings and financial news.</li>\
      <li>End-to-end machine learning workflows with CI/CD integration, including data preprocessing, model training, and deployment, leveraging SageMaker Pipelines for efficient, scalable, and automated ML operations.</li></ul>",
    Year: "2024",
    Date: "Feb 21, 2024",
    Skills: ["Python", "AWS", "AWS SageMaker", "Hugging Face", "Docker", "CI/CD", "NLP", "Sentiment Analysis", "MLOps",],
    TeamSize: 1,
    GithubLink: "",
  },

  {
    Title: "Decades of Cinema Insight",
    Description:
      "<ul class='project-ul'><li>Built a web application analyzing movie trends (1986-2016) with a dataset of 6,820 films, focusing on factors like budget, genre, revenue, and ratings to determine success, featuring trend visualization and correlation tools.</li>\
      <li>Led the integration of React for frontend development, FastAPI and MongoDB for backend processing, and deployment on Render, ensuring a responsive user experience and seamless data management.</li></ul>",
    Year: "2023",
    Date: "Nov 30, 2023",
    Skills: ["React", "FastAPI", "MongoDB", "nosql", "Full-Stack Development",],
    TeamSize: 3,
    GithubLink: "https://github.com/santhu932/Movie-Analysis",
  },

  {
    Title: "Global News Analysis for Stock Prediction",
    Description:
      "Employed financial-BERT, which surpassed traditional statistical models by 26% in accuracy, to enhance stock market predictions through analysis of ‘r/worldnews’ subreddit data, encompassing global news and economic events.",
    Year: "2023",
    Date: "Oct 24, 2023",
    Skills: [
      "Pytorch",
      "NLTK",
      "Pandas",
      "Scikit-learn",
      "Lemmatization",
      "Count vectorization",
      "Large Language Models",
      "Seaborn",
    ],
    TeamSize: 2,
    GithubLink: "https://github.com/santhu932/Stock-Market-Prediction-using-NLP",
  },

  
  {
    Title: "Embedded Topic Modeling (ETM)",
    Description:
      "<ul class='project-ul'><li>Implemented topic modeling in embedding spaces by leveraging word embedding and assigned topic embedding, resulting in enhanced representation of topic distribution and incorporated Variational Autoencoder to approximate the posterior for improved model fit.</li>\
      <li>Evaluated the effectiveness of the LDA Model against the ETM Model, employing rigorous metrics such as Topic Perplexity and Topic Diversity to identify the model that yielded superior results in accurate topic extraction.</li></ul>",
    Year: "2023",
    Date: "Apr 30, 2023",
    Skills: ["Topic Modeling", "Pytorch", "Natural Language Processing", "Variational Autoencoders"],
    TeamSize: 1,
    GithubLink:
      "https://github.com/santhu932/Topic-Modeling-ETM",
  },
  {
    Title: "General Detection of Image Manipulation",
    Description:
      "Engineered a cutting-edge deep learning model by integrating Error Level Analysis and ResNeXt, enabling highly accurate detection of image manipulation; outperformed industry benchmarks and established new standards in image forensics with 94% accuracy score.",
    Year: "2023",
    Date: "Apr 25, 2023",
    Skills: ["Computer Vision", "PyTorch", "Error level analysis", "Deep learning", "OpenCV", "Pillow"],
    TeamSize: 4,
    GithubLink:
      "https://github.com/santhu932/Image-Manipulation-Detection",
  },

  {
    Title: "Normalizing Flows in 2D",
    Description:
      "Implemented normalizing flows (Forward, Backward Flows and Density Estimation) using automatic differentiation for the optimization that utilizes Black Box Variational Inference with reparameterization",
    Year: "2023",
    Date: "Mar 30, 2023",
    Skills: [
      "Pytorch",
      "BBVI-Reparameterization",
      "Normalizing Flows",
      "Variational Inference",
      "Approximate Inference in Graphical Models"
    ],
    TeamSize: 1,
    GithubLink: "https://github.com/santhu932/Normalizing-Flows",
  },

  {
    Title: "Bayesian approach to Movie Recommendation Model",
    Description:
      "<ul class='project-ul'><li>Analyzed and restructured Variational Probabilistic Matrix Factorization Algorithm presented in the paper “Variational Bayesian approach to Movie Rating Prediction”.</li>\
      <li>Trained Variational Expectation-Maximization Algorithm having the update equations obtained from Mean Field Approximation for E-step on data set containing 100k observed values in the matrix.</li>\
      <li>Assessed RMSE discrepancies between baseline and alternative initialization approach for decomposed matrices U and V, leveraging values from the initial iteration of EM-SVD(Singular Value Decomposition) to achieve a 10% increase in model convergence.</li></ul>",
    Year: "2023",
    Date: "Feb 25, 2023",
    Skills: [
      "Python",
      "Numpy",
      "SciPy",
      "Matplotlib",
      "Singular Value Decomposition",
      "Variational Inference",
      "Matrix Factorization",
      "EM Algorithm",
      "Mean-Field Approximation",
    ],
    TeamSize: 1,
    GithubLink: "https://github.com/santhu932/Movie-Rating-Prediction",
  },

  {
    Title: "Projections, Transformations, Cameras, Stereo",
    Description:
      '<ul class="project-ul"><li>Developed 3D-to-2D projection task that computes projection matrices to simulate flight of an airplane.</li>\
      <li>Implemented Loopy Belief Propagation to address complex optimization problems that involved processing political data to find optimal solutions under certain constraints.</li>\
      <li>Designed a stereo matching algorithm that tackled the challenge of estimating disparities between stereo images which utilizes a more sophisticated Loopy Belief Propagation technique.</li></ul>',
    Year: "2023",
    Date: "Mar 25, 2023",
    Skills: ["Python", "Pillow", "Trnasformations and Projections", "Computer Vision", "Depth Estimation", "Markov Random Fields", "Loopy Belief Propagation"],
    TeamSize: 4,
    GithubLink: "https://github.com/santhu932/Computer-Vision-Challenges/tree/main/Projections%2C%20Transformations%2C%20Cameras%2C%20Stereo",
  },

  {
    Title: "Optical Music Recognition",
    Description:
      "<ul class='project-ul'><li>Built a model to detect staff lines, notes and rests from the music sheet. Identified staffs using Hough transform with at least 70% votes for the lines in accumulator gathered from Hough space.</li>\
      <li>Designed a Normalized Cross-Correlation template matching algorithm to identify bounding boxes for notes, eighth notes, and quarter rests using specific thresholds: k = 4, 1.75, and 2, respectively.</li></ul>",
    Year: "2023",
    Date: "Feb 18, 2023",
    Skills: ["Python", "Pillow", "Hough Transform", "Computer Vision", "Object Detection",],
    TeamSize: 3,
    ProfessorLink: "",
    GithubLink: "https://github.com/santhu932/Computer-Vision-Challenges/tree/main/Optical%20Music%20Recognition",
  },

  {
    Title: "B505: Applied Algorithm",
    Description:
      "Solved various DSA problems as a part of assignments",
    Year: "2023",
    Date: "Jan 30, 2023",
    Skills: ["Data Structures: Stacks, Queues, Trees, Graphs, Linked lists, Arrays, Set", "Algorithms: Dynamic Programming, Backtracking, DFS, BFS, Binary Search, Sliding Window",],
    TeamSize: 1,
    GithubLink:
      "https://github.com/santhu932/DSA-Problems",
  },

  {
    Title: "Topic Model (LDA Algorithm)",
    Description:
      "Implemented the collapsed Gibbs Sampler for LDA inference on '20newsgroups' dataset and assessed the LDA topic representation to a “bag-of-words” representation with respect to how well models support document classification.",
    Year: "2022",
    Date: "Nov 15, 2022",
    Skills: ["Python", "Numpy", "Topic Modeling", "Gibbs Sampling",],
    TeamSize: 1,
    GithubLink: "",
  },

  {
    Title: "Experiments on classification models",
    Description:
      "<ul class='project-ul'><li>Implemented and assessed two binary classification algorithms: two-class generative model with a shared covariance matrix, and Bayesian logistic regression.</li>\
      <li>Compared two optimization methods for above models: Newton's method and Gradient Ascent.</li></ul>",
    Year: "2022",
    Date: "Oct 30, 2022",
    Skills: [
      "Python",
      "Numpy",
      "Bayesian Inference",
      "Optimization Methods",
      "Generative Models",
      "Discriminative Models",
    ],
    TeamSize: 1,
    GithubLink: "",
  },

  {
    Title: "B561: Advanced Database Concepts",
    Description:
      "Worked on assignments which involved solving advanced SQL queries, eelational algebra Expressions and query optimization problems. ",
    Year: "2022",
    Date: "Sept 25, 2022",
    Skills: [
      "SQL-Joins, Semijoins, Views",
      "Tuple Relational Calculus",
      "Relational Algebra",
    ],
    TeamSize: 1,
    GithubLink: "",
  },

  {
    Title: "Unigram Model (Hyperparameter Tuning using Evidence Maximization)",
    Description:
      "<ul class='project-ul'><li>Implemented Unigram Model to get a probability distribution over vocabulary of words using Maximum Likelihood estimate, MAP estimate and Predictive distribution, and compared the perplexity of the learned models.</li>\
      <li>Hyper-parameters were determined by maximizing the evidence function (Model Selection algorithm). Evidence approximation significantly sped up finding optimal parameters compared to grid search.</li></ul>",
    Year: "2022",
    Date: "Sept 30, 2022",
    Skills: [
      "Python",
      "Numpy",
      "Matplotlib",
      "Hyperparamter Tuning",
      "EM Algorithm",
      "Bayesian Inference",
    ],
    TeamSize: 1,
    GithubLink: "",
  },

  {
    Title: "Polynomial Curve Fitting (Experiments with Bayesian Linear Regression)",
    Description:
      "<ul class='project-ul'><li>Compared the non-regularized linear regression and Bayesian model over the polynomial degrees (d=1,2,3…). Hyper-parameters were determined using bayesian model selection.</li>\
      <li>Evidence function was used to select degree in case of bayesian method.</li></ul>",
    Year: "2022",
    Date: "Aug 30, 2022",
    Skills: ["Python", "Numpy", "Bayesian Linear Regression", "Probabilistic Models",],
    TeamSize: 1,
    GithubLink: "",
  },

  {
    Title: "Breast Cancer Detection",
    Description:
      "Designed a CNN architecture to identify tumor cells and classify the tissue cells as either malignant or benign",
    Year: "2019",
    Date: "Nov 30, 2019",
    Skills: [
      "Pytorch",
      "Python",
      "Matplotlib",
      "OpenCV",
      "Neural Networks",
    ],
    TeamSize: 2,
    GithubLink: "",
    TitleLink: "",
  },
];

class MyHeader extends HTMLElement {
  static observedAttributes = ["pageTitle", "theme"];

  constructor() {
    super();
    this.theme = "light";
    this.pageTitle = this.getAttribute("pageTitle");
  }

  get theme() {
    return this.getAttribute("theme");
  }

  set theme(newValue) {
    this.setAttribute("theme", newValue);
  }

  attributeChangedCallback(attr, oldValue, newValue) {
    if (oldValue !== newValue && attr === "pageTitle") {
      this.pageTitle = newValue;
    } else if (oldValue !== newValue && attr === "theme") this.theme = newValue;
  }

  connectedCallback() {
    this.setInitialTheme();
    this.render();
    this.querySelector("#theme-toggle").addEventListener(
      "click",
      this.onThemeChange
    );
  }

  setInitialTheme() {
    const initialTheme = localStorage.getItem("theme");
    const setInitialTheme = initialTheme === null ? "light" : initialTheme;

    this.theme = setInitialTheme;

    if (setInitialTheme === "light")
      document.body.classList.remove("dark-theme");
    else document.body.classList.add("dark-theme");
  }

  onThemeChange() {
    const currentTheme = localStorage.getItem("theme");
    const newTheme = currentTheme === "light" ? "dark" : "light";

    this.theme = newTheme;

    if (newTheme === "light") document.body.classList.remove("dark-theme");
    else document.body.classList.add("dark-theme");
    localStorage.setItem("theme", newTheme);
  }

  renderThemeIcon() {
    const lightIcon = `<svg id="theme-toggle" width="16" height="16" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M12 18.14a.722.722 0 0 0-.722.722v2.166a.722.722 0 0 0 1.444 0v-2.166a.722.722 0 0 0-.722-.721ZM12 2.25a.722.722 0 0 0-.722.722v2.166a.722.722 0 0 0 1.444 0V2.972A.722.722 0 0 0 12 2.25ZM5.86 12a.722.722 0 0 0-.723-.722H2.972a.722.722 0 0 0 0 1.444h2.165A.722.722 0 0 0 5.86 12ZM21.028 11.278h-2.165a.722.722 0 0 0 0 1.444h2.165a.722.722 0 0 0 0-1.444ZM7.148 16.13a.72.72 0 0 0-.51.21l-1.533 1.534a.72.72 0 1 0 1.022 1.022l1.532-1.533a.724.724 0 0 0-.51-1.233ZM16.852 7.87a.72.72 0 0 0 .51-.21l1.533-1.533a.72.72 0 0 0 .211-.511.72.72 0 0 0-.722-.722.72.72 0 0 0-.51.21L16.34 6.639a.72.72 0 0 0-.211.51.72.72 0 0 0 .722.722ZM6.127 5.105a.72.72 0 0 0-.511-.211.72.72 0 0 0-.722.722.72.72 0 0 0 .21.51L6.638 7.66a.72.72 0 0 0 .511.211.72.72 0 0 0 .722-.722.72.72 0 0 0-.21-.51L6.126 5.105ZM17.363 16.34a.72.72 0 1 0-1.022 1.022l1.532 1.533a.72.72 0 0 0 1.022 0 .72.72 0 0 0 0-1.021l-1.532-1.533ZM12 7.5c-2.48 0-4.5 2.02-4.5 4.5s2.02 4.5 4.5 4.5 4.5-2.02 4.5-4.5-2.02-4.5-4.5-4.5Z" fill="currentColor"></path></svg>`;
    const darkIcon = `<svg id="theme-toggle" width="16" height="16" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M20.742 13.045a8.086 8.086 0 0 1-2.077.271c-2.135 0-4.14-.83-5.646-2.336a8.026 8.026 0 0 1-2.064-7.723A1 1 0 0 0 9.73 2.034a10.014 10.014 0 0 0-4.489 2.582c-3.898 3.898-3.898 10.243 0 14.143a9.936 9.936 0 0 0 7.072 2.93 9.93 9.93 0 0 0 7.07-2.929 10.007 10.007 0 0 0 2.583-4.491 1 1 0 0 0-1.224-1.224Zm-2.772 4.301a7.947 7.947 0 0 1-5.656 2.343 7.952 7.952 0 0 1-5.658-2.344c-3.118-3.119-3.118-8.195 0-11.314a7.923 7.923 0 0 1 2.06-1.483 10.027 10.027 0 0 0 2.89 7.848 9.973 9.973 0 0 0 7.848 2.891 8.037 8.037 0 0 1-1.484 2.059Z" fill="currentColor"></path></svg>`;

    return this.theme === "dark" ? lightIcon : darkIcon;
  }

  render() {
    this.innerHTML = `
      <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta http-equiv="X-UA-Compatible" content="ie=edge">
        <link rel="icon" type="image/x-icon" href="favicon.ico">
        <link rel="stylesheet" href="./style.css">
        <title>Santhosh M Patil: ${this.pageTitle}</title>
      </head>

      <header>
        <div class="header-container">
          <h1 class="name">Santhosh M Patil</h1>
          <div title="Toggle Theme" class="theme-icon">
            ${this.renderThemeIcon()}
          </div>
        </div>
        <hr>
        <div class="row-header">
          <section class="row-header-item" style="padding-left: 0;" title="Home"><a href="./index.html">Home</a></section>
          <section class="row-header-item" title="Experience"><a href="./experience.html">Experience</a></section>
          <section class="row-header-item" title="Education"><a href="./education.html">Education</a></section>
          <section class="row-header-item" title="Projects"><a href="./projects.html">Projects</a></section>
          <section class="row-header-item" title="Publications"><a href="./publications.html">Publications/Conferences</a></section>
          <section class="row-header-item" style="padding-right: 0;" title="Contact"><a href="./contact.html">Contact</a></section>
        </div>
        <h2 style="padding-top: 12px;">${this.pageTitle}</h2>
      </header>`;
  }
}

class MyHome extends HTMLElement {
  constructor() {
    super();
  }

  connectedCallback() {
    this.innerHTML = `
      <div class="page-content">
        <div class="profile-container">
          <img class="profile-image" src="./assets/profile-image.jpg" alt="Profile Picture of Santhosh">
          <div>
            <strong>Santhosh Manohar Gouda Patil</strong>
            <section class="small-pad" style="margin-top: 8px;">Graduate Student & Research Assistant</section>
            <section class="small-pad">Indiana University, Bloomington</section>
            <br>
            <section class="small-pad"><a href="mailto://sanpati@iu.edu">sanpati@iu.edu</a></section>
            <section class="small-pad">Bloomington, Indiana, USA - 47401</section>
            <br>
          </div>
        </div>
        <div class="about-text">
          <h4>About Me</h4>
          <p>
            I'm pursuing masters in Data Science at the
            <a target="_blank" href="https://luddy.indiana.edu/index.html">Luddy School of Informatics, Computing, and Engineering</a>,
            <a target="_blank" href="https://bloomington.iu.edu/index.html">Indiana University Bloomington</a>.
            My research interest lies in probabilistic machine learning, computer vision, time series analysis and NLP. Under the
            guidance of <a target="_blank" href="https://cgi.luddy.indiana.edu/~rkhardon/">
              Prof. Roni Khardon</a>,
            my research focuses on probabilistic deep learning forecasting models and its application in weather prediction.
          </p>
          <p>
            I previously worked as a associate developer at <a href="https://www.ibm.com/">IBM</a>
            in Bengaluru, India, where I developed data retention policies for the Salesforce platform and implemented a custom churn prediction model adhering to industry standards. 
            Prior to that, I served as a research intern at the <a href="https://iisc.ac.in">Indian Institute of Science</a> under the guidance of <a target="_blank" href="https://scholar.google.co.in/citations?user=OftxRCEAAAAJ&hl=en">
            Prof. Raghu Krishnapuram</a>. During my internship, I focused on semantic segmentation, depth estimation, and visual odometry for autonomous navigation.
          </p>
          <p>
          I completed my Bachelor’s degree in Computer Science at
            <a target="_blank" href="https://rvce.edu.in">R. V. College of Engineering (RVCE)</a> in
            Bangaluru, India. During my studies, I collaborated with Prof. Rajashree Shettar on projects focused on object recognition and classification in autonomous vehicles. I also developed a deep learning-based auto-annotation tool as part of a joint initiative involving
            <a target="_blank" href="https://rvce.edu.in">RVCE</a>, <a target="_blank" href="https://iisc.ac.in">IISc</a> and <a target="_blank" href="https://www.wipro.com">WIPRO</a>. In addition, I volunteered with the Rotaract Club at RVCE.
          </p>
        </div>
      </div>`;
  }
}

class MyEducation extends HTMLElement {
  constructor() {
    super();
  }

  connectedCallback() {
    this.innerHTML = `
      <div id="education">
        <div class="education-section" style="margin-top: -8px;">
          <div class="education-text">
            <h3 id="masters">Luddy School of Informatics, Computing, and Engineering, Indiana University, Bloomington</h3>
            <section class="degree-title">Master of Science, Data Science</section>
            <section class="education-subtitle">
              Aug 2022 - May 2024
              <br>
              Bloomington, Indiana, USA
              <br>
              <br>
              sanpati@iu.edu
            </section>
          </div>
          <figure class="education-figure-container">
            <img class="education-image" src="./assets/iub.jpeg" alt="Indiana University Bloomington.">
            <figcaption>Sample Gates</figcaption>
          </figure>
        </div>
        <br>
        <br>
        <div class="education-section">
          <div class="education-text">
            <h3 id="bachelors">R. V. College of Engineering (RVCE)</h3>
            <section class="degree-title">Bachelor of Engineering, Computer Science & Engineering</section>
            <section class="education-subtitle">
              Aug 2016 - July 2020
              <br>
              Mysore Road, Bangaluru, Karnataka, India
              <br>
              <br>
              santhoshmpatil.cs16@rvce.edu.in
            </section>
          </div>
          <figure class="education-figure-container">
            <img class="education-image" src="./assets/rvce.jpg" alt="RVCE Front Entrance.">
            <figcaption>RVCE Front Gate</figcaption>
          </figure>
        </div>
        <br>
        <br>
      </div>`;
  }
}

class MyExperience extends HTMLElement {
  constructor() {
    super();
  }

  connectedCallback() {
    this.innerHTML = `
      <div style="font-size: 14px;">
        <a href="#current">Current</a>&nbsp; |&nbsp;
        <a href="#research">Research</a>&nbsp; |&nbsp;
        <a href="#previous">Previous</a>&nbsp; |&nbsp;
        <a href="#internships">Internships</a>
      </div>
      <br>
      <div id="current">
        <section class="experience-title">Current</section>
      </div>
      <hr class="hr">
      <div id="research">
        <section class="experience-title">Research</section>
        <section class="company-title">Indiana University, Bloomington</section>
        <section class="role-title">Graduate Research Assistant</section>
        <section style="padding: 8px 0 0 0;">Prof. Roni Khardon</section>
        <ul>
          <li>
          Led an innovative research project that integrates Convolutional LSTM and Transformer models to enhance the forecasting of tropical
          cyclone genesis in the Northern Pacific.</li>
          <li>
          Optimized Earthformer model's hierarchical encoder-decoder structure for accurate atmospheric predictions on tropical cyclone formation,
          integrating cuboid attention and global vectors to achieve a 15% drop in RMSE.</li>
          <li>
          Implemented a Convolutional LSTM model to identify TC locations, enhancing Nowcasting capabilities through convolutional input-to-state
          and state-to-state transitions, achieving a 0.82 F1 score.</li>
          <li>
          Crafted an efficient, high-accuracy Ensemble model with multiprocessing and DDP on Multi-GPU, reaching a 0.75 score for 42-hour forecasts.
          </li>
          <li>
          Explored probabilistic Bayesian approaches for better uncertainty estimation and to reduce computational costs compared to ensembles.
          </li>
          <li>
          Improved forecasting precision by 17% through refinement of weather variables such as wind speed and temperature at different sea levels;
          employed a novel Masked-Minmax normalization technique for effective training and informed decision-making.
          </li>
        </ul>
      </div>
      <!-- <hr class="hr"> -->
      <hr class="hr">
      <div id="previous">
        <section class="experience-title">Previous</section>
        <div class="company" id="ibm">
          <div>
            <section class="company-title">IBM</section>
            Jan 2021 - June 2022
            <br>
            <br>
            G2 Blook
            <br>
            Manyata Tech Park, Bangaluru, India
            <br>
            <br>
            <section class="role-title">Associate Developer</section>
            <ul>
              <li>Developed and deployed an XGBoost-based churn prediction model using AWS SageMaker. Utilized AWS Glue for ETL processes and data
              integration from Salesforce, enhancing predictive analytics by 13%.
              </li>
              <li>Conducted comparative experiments on XGBoost and Random Forest algorithms to assess model accuracy and robustness, selecting XGBoost
              for deployment due to its higher efficiency and scalability in churn prediction.
              </li>
              <li>Developed a secure data retention policy to protect production data, leveraging SQL for effective data handling and compliance, minimizing
              unauthorized access risks.
              </li>
            </ul>
          </div>
        </div>
      </div>
      <hr class="hr">
      <div id="internships">
        <section class="experience-title">Internships</section>
        <section class="company-title" id="iisc-internship">Robert Bosch Centre for Cyber Physical Systems, Indian Institute of Science</section>
        Oct 2019 - Jun 2020
        <br>
        <br>
        <section class="role-title">Research Intern: Prof. Raghu Krishnapuram</section>
        <ul>
          <li>
          Crafted a sophisticated U-Net-based deep learning model for monocular depth estimation, incorporating innovative loss functions, which
          led to a 5% decrease in RMSE and enhanced depth map quality on the KITTI benchmark, outperforming prior models.
          </li>
          <li>
          Pioneered efficient training methods with multi-scale sampling and a distinct pose estimation network for self-supervised learning,
          eliminating the need for ground truth depth.
          </li>
          <li>
          Led the exploration of Visual Odometry using Deep Recurrent CNNs, enhancing image sequence modeling for autonomous navigation; results
          demonstrated a 12% increase in accuracy for real-time pose estimation.</li>
          <li>Optimized the Mask R-CNN semantic segmentation model, enhancing accuracy in object detection and classification within intricate visual
          settings, achieving a 20% improvement in model precision.</li>
        </ul>
        
        <section class="company-title" id="iisc-internship">Indian Institute of Science (WIRIN - Wipro IISc Innovation Network)</section>
        Jun 2019 - Aug 2019
        <br>
        <br>
        <section class="role-title">Research Intern: Dr. Ramachandra budihal and Prof. Rajashree Shettar</section>
        <ul>
          <li>
          Expanded the capabilities of the PolygonRNN Auto - Annotation model by incorporating the YOLO Algorithm for precise object
          recognition and classification, accelerating annotation speed by 40% and elevating overall annotation quality.
          </li>
        </ul>
      </div>`;
  }
}

class MyFilter extends HTMLElement {
  constructor() {
    super();
    this.skills = [];
    this.years = [];
    this.titles = [];
    this.selectedSkills = [];
    this.selectedYears = [];
    this.selectedTitles = [];
    this.projects = window.projects;
  }

  connectedCallback() {
    this.prepareFilters();
    // FIXME When filter is added clicking on x of filter must also trigger this event.
    this.addEventListener("click", this.onFilterSelect);
    this.render();
  }

  prepareFilters() {
    this.years = this.projects.map((p) => p.Year);
    this.years = [...new Set(this.years)]; // Remove duplicates
    // this.years = this.years.filter((value, index)=> this.years.indexOf(value) === index); // Remove duplicates
    this.years.sort().reverse(); // Current year at the beginning.

    this.titles = this.projects.map((p) => p.Title);
    this.titles = [...new Set(this.titles)]; // Remove duplicates

    // this.professors = this.projects.map((p) => p.Professor);
    // this.professors = [...new Set(this.professors)]; // Remove duplicates

    this.skills = this.projects.map((p) => p.Skills).flat(1); // Convert 2D array to 1D array.
    this.skills = [...new Set(this.skills)]; // Remove duplicates
    this.skills.sort();
  }

  onFilterSelect() {
    let selectedYears = Array.from(
      document.getElementById("selectedYears").options
    )
      .filter((option) => option.selected)
      .map((option) => option.value);

    // console.log("this.selectedYears", selectedYears);
    this.selectedYears = selectedYears;

    let selectedSkills = Array.from(
      document.getElementById("selectedSkills").options
    )
      .filter((option) => option.selected)
      .map((option) => option.value);

    // console.log("this.selectedSkills", selectedSkills);
    this.selectedSkills = selectedSkills;

    let selectedTitles = Array.from(
      document.getElementById("selectedTitles").options
    )
      .filter((option) => option.selected)
      .map((option) => option.value);

    // console.log("this.selectedTitles", selectedTitles);
    this.selectedTitles = selectedTitles;

    // Prepare the projects to show with filters selected.
    // Start from all projects and filter one bye one.
    this.selectedProjects = this.projects;
    if (this.selectedYears.length) {
      this.selectedProjects = this.selectedProjects.filter((p) =>
        this.selectedYears.includes(p.Year)
      );
    }

    // console.log("this.selectedProjects", this.selectedProjects);
    if (this.selectedSkills.length) {
      this.selectedProjects = this.selectedProjects.filter((project) =>
        project.Skills.some((skill, _) => this.selectedSkills.includes(skill))
      );
    }

    if (this.selectedTitles.length) {
      this.selectedProjects = this.selectedProjects.filter((p) =>
        this.selectedTitles.includes(p.Title)
      );
    }

    this.renderMyProject();
  }

  renderMyProject() {
    const component = document.querySelector("my-project");
    component.projects = JSON.stringify(this.selectedProjects);
  }

  renderFilters() {
    let filters = ``;
    const filterIcon = `<svg class="filter-icon" xmlns="http://www.w3.org/2000/svg" data-name="Layer 2" viewBox="0 0 30 30" id="filter"><path fill="#111224" d="M17 11H4A1 1 0 0 1 4 9H17A1 1 0 0 1 17 11zM26 11H22a1 1 0 0 1 0-2h4A1 1 0 0 1 26 11z"></path><path fill="#111224" d="M19.5 13.5A3.5 3.5 0 1 1 23 10 3.5 3.5 0 0 1 19.5 13.5zm0-5A1.5 1.5 0 1 0 21 10 1.5 1.5 0 0 0 19.5 8.5zM26 21H13a1 1 0 0 1 0-2H26A1 1 0 0 1 26 21zM8 21H4a1 1 0 0 1 0-2H8A1 1 0 0 1 8 21z"></path><path fill="#111224" d="M10.5,23.5A3.5,3.5,0,1,1,14,20,3.5,3.5,0,0,1,10.5,23.5Zm0-5A1.5,1.5,0,1,0,12,20,1.5,1.5,0,0,0,10.5,18.5Z"></path></svg>`;

    let yearFilter = `<select id="selectedYears" placeholder="Select Year" txtSearch="Search Year" style="width: 20%" multiple multiselect-search="true" @click=${this.onFilterSelect}>`;
    yearFilter += this.years.map(
      (year) => `<option value="${year}">${year}</option>`
    );
    yearFilter += "</select>";

    let skillFilter = `<select id="selectedSkills" placeholder="Select Skills" txtSearch="Search Skills" style="width: 25%" multiple multiselect-search="true" @click=${this.onFilterSelect}>`;
    skillFilter += this.skills.map(
      (skill) => `<option value="${skill}">${skill}</option>`
    );
    skillFilter += "</select>";

    let titleFilter = `<select id="selectedTitles" placeholder="Select Projects" txtSearch="Search Projects" style="width: 35%" multiple multiselect-search="true" @click=${this.onFilterSelect}>`;
    titleFilter += this.titles.map(
      (skill) => `<option value="${skill}">${skill}</option>`
    );
    titleFilter += "</select>";

    filters += `<div class="filters-div">${filterIcon} ${titleFilter}&nbsp; ${skillFilter}&nbsp; ${yearFilter}</div>`;
    return filters;
  }

  render() {
    this.innerHTML = `${this.renderFilters()}`;
  }
}

class MyProject extends HTMLElement {
  static observedAttributes = ["projects"];

  constructor() {
    super();
    this.skillCount = 0;
    this.totalSkillCount = this.getSkillCount(window.projects);
    this.projectCount = window.projects.length;
    // Property projects here is a string. my-filter component sends a string of objects.
    this.projects = JSON.stringify(window.projects);
  }

  connectedCallback() {
    if (!this.projects) return;

    let projectMap = this.prepareProjectMap();
    this.render(projectMap);
  }

  getSkillCount(projects) {
    let skillCount;
    skillCount = projects.map((p) => p.Skills).flat(1); // Convert 2D array to 1D array.
    skillCount = [...new Set(skillCount)]; // Remove duplicates
    return skillCount.length;
  }

  get projects() {
    return this.getAttribute("projects");
  }

  set projects(newValue) {
    this.setAttribute("projects", newValue);
  }

  attributeChangedCallback(_, oldValue, newValue) {
    // Don't change the project value when newValue is empty.
    if (newValue && oldValue !== newValue) {
      this.projects = newValue;
      this.connectedCallback(); // Re-render manually when the project value changes.
    }
  }

  prepareProjectMap() {
    // Convert the string from my-filter to object.
    let projects = JSON.parse(this.projects);
    this.projectCount = projects.length;
    this.skillCount = this.getSkillCount(projects);

    let projectMap = projects.reduce(function (map, project) {
      if (!(project.Year in map)) map[project.Year] = [];
      map[project.Year].push(project);
      return map;
    }, {});

    const map = new Map(Object.entries(projectMap));
    let sortedArray = Array.from(map.entries());

    sortedArray.sort((a, b) => b[0] - a[0]);
    let sortedMap = new Map(sortedArray);

    return sortedMap;
  }

  getProjectTitle(p) {
    if (p.TitleLink && p.TitleLink !== "")
      return `<h3><a target="_blank" href="${p.TitleLink}">${p.Title}</a></h3>`;

    return `<h3>${p.Title}</h3>`;
  }

  getGithubLink(p) {
    if (p.GithubLink && p.GithubLink !== "")
      return `
        <section class="project-subsection">
          Code:&nbsp;
          <a target="_blank" href="${p.GithubLink}">
            ${p.GithubLink.replace("https://", "")}
          </a>
        </section>`;

    return ``;
  }

  getTeamSize(p) {
    if (p.TeamSize && p.TeamSize > 1)
      return `&nbsp;•&nbsp;Team of ${p.TeamSize}`;

    return ``;
  }

  getProfessorName(p) {
    if (p.Professor && p.Professor !== "")
      return `&nbsp;•&nbsp; Taught by ${p.Professor}`;

    return ``;
  }

  getSkills(p) {
    if (p.Skills && p.Skills.length) {
      return `&nbsp;•&nbsp; Skills:${p.Skills.map(
        (tag) => `&nbsp;&nbsp;#${tag}`
      ).join("")}`;
    }

    return ``;
  }

  renderProject(p) {
    return `
      ${this.getProjectTitle(p)}
      <section class="project-subsection">
        ${p.Date} ${this.getSkills(p)}
        ${this.getTeamSize(p)} ${this.getProfessorName(p)}
      </section>
      ${this.getGithubLink(p)}
      <section class="project-description">${p.Description}</section>
      <hr class="hr">`;
  }

  renderProjectByYear(projects) {
    return projects.map((p) => `${this.renderProject(p)}`).join(" ");
  }

  render(projectMap) {
    let resultantHTML = `<section class="filter-result">Showing ${this.projectCount} projects from ${window.projects.length} projects with ${this.skillCount} skills from ${this.totalSkillCount} skills.</section>`;

    projectMap.forEach((projects, year) => {
      resultantHTML += `<h2>${year}</h2>${this.renderProjectByYear(projects)}`;
    });

    this.innerHTML = resultantHTML;
  }
}

class MyPublication extends HTMLElement {
  constructor() {
    super();
  }

  connectedCallback() {
    this.innerHTML = `
      <div id="#publications">
        <ol>
          <li>
            <section class="paper-title">
              <a target="_blank" href="https://ams.confex.com/ams/36Hurricanes/meetingapp.cgi/Paper/442094">
              Predictability of Tropical Cyclone Formation with Large-Scale Memory Using Deep Learning Transformer
              </a>
            </section>
            Our research work is being presented at the American Metrological Society, Conference on Hurricanes and Tropical Meteorology, May, 2024: <a target="_blank" href="https://ams.confex.com/ams/36Hurricanes/meetingapp.cgi/Paper/442094">Abstract</a>.
            <section class="paper-authors">
              <i>Yadi Wei, <strong>Santhosh M Patil</strong>, Roni Khardon, Chanh Kieu
              </i>
            </section>
          </li>
        </ol>
      </div>`;
  }
}

class MyContact extends HTMLElement {
  constructor() {
    super();
  }

  connectedCallback() {
    this.innerHTML = `
      <div id="contact">
        <p style="padding: 6px 0 0 0;">Hello there!</p>
        <p>Feel free to reach out to me on any of the platforms.</p>
        <ul>
          <li><a target="_blank" href="mailto:sanpati@iu.edu">sanpati@iu.edu</a></li>
          <li><a target="_blank" href="https://github.com/santhu932">github.com/santhu932</a></li>
          <li>
            <a target="_blank" href="https://www.linkedin.com/in/santhosh-m-patil/">linkedin.com/in/santhosh-m-patil</a>
          </li>
          <li>
            <a target="_blank" href="https://medium.com/@santhoshpatil932">medium.com/@santhoshpatil932</a>
          </li>
        </ul>
        <p>Thanks for stopping by :)</p>
      </div>`;
  }
}

class MyFooter extends HTMLElement {
  static observedAttributes = [
    "showHR",
    "showTop",
    "showCopyright",
    "showLastUpdated",
  ];

  constructor() {
    super();
    this.lastUpdated = "Apr 10, 2024";
    this.showHR = this.hasAttribute("showHR");
    this.showTop = this.hasAttribute("showTop");
    this.showCopyright = this.hasAttribute("showCopyright");
    this.showLastUpdated = this.hasAttribute("showLastUpdated");
  }

  connectedCallback() {
    const lastUpdated = `<div class="footer-text">Last Updated ${this.lastUpdated}</div>`;
    const copyright = `<div class="footer-text">Template inspired by my friend Muteeb Akram.</div>`;

    this.innerHTML = `
      ${this.showHR ? `<hr>` : ``}
      <footer>
        <div class="footer-container">
          ${this.showCopyright ? copyright : ``}
          ${this.showTop ? `` : ``}
          ${this.showLastUpdated ? lastUpdated : ``}
        </div>
      </footer>`;
  }
}

customElements.define("my-header", MyHeader);
customElements.define("my-home", MyHome);
customElements.define("my-experience", MyExperience);
customElements.define("my-education", MyEducation);
customElements.define("my-publication", MyPublication);
customElements.define("my-filter", MyFilter);
customElements.define("my-project", MyProject);
customElements.define("my-contact", MyContact);
customElements.define("my-footer", MyFooter);
