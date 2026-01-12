#include "create_project_dialog.h"
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QVBoxLayout>

CreateProjectDialog::CreateProjectDialog(QWidget *parent)
    : QDialog(parent) {
  setWindowTitle("Create New Project");
  setFixedSize(500, 450);
  setupUI();
}

void CreateProjectDialog::setupUI() {
  auto *mainLayout = new QVBoxLayout(this);

  // Project name
  auto *projectNameLayout = new QHBoxLayout();
  auto *projectNameLabel = new QLabel("Project Name:", this);
  m_projectNameEdit = new QLineEdit(this);
  projectNameLayout->addWidget(projectNameLabel);
  projectNameLayout->addWidget(m_projectNameEdit);
  mainLayout->addLayout(projectNameLayout);

  // Source type selection
  auto *sourceGroup = new QGroupBox("Project Source", this);
  auto *sourceLayout = new QVBoxLayout(sourceGroup);

  m_localRadio = new QRadioButton("Local File", sourceGroup);
  m_serverRadio = new QRadioButton("Server URL", sourceGroup);
  m_localRadio->setChecked(true);

  sourceLayout->addWidget(m_localRadio);
  sourceLayout->addWidget(m_serverRadio);

  // Local file section
  m_localWidget = new QWidget(sourceGroup);
  auto *localLayout = new QHBoxLayout(m_localWidget);
  localLayout->setContentsMargins(0, 0, 0, 0);

  m_pathLabel = new QLabel("Original Path:", sourceGroup);
  m_originalPathEdit = new QLineEdit(m_localWidget);
  m_browseButton = new QPushButton("Browse...", m_localWidget);

  localLayout->addWidget(m_originalPathEdit);
  localLayout->addWidget(m_browseButton);

  // Server URL section
  m_serverWidget = new QWidget(sourceGroup);
  auto *serverLayout = new QVBoxLayout(m_serverWidget);
  serverLayout->setContentsMargins(0, 0, 0, 0);

  m_urlLabel = new QLabel("Server URL:", sourceGroup);
  m_serverUrlEdit = new QLineEdit(m_serverWidget);
  m_serverUrlEdit->setPlaceholderText("http://example.com/sparql");

  auto *serverUrlLayout = new QHBoxLayout();
  serverUrlLayout->addWidget(m_serverUrlEdit);
  serverLayout->addLayout(serverUrlLayout);

  m_graphUriLabel = new QLabel("Graph URI:", sourceGroup);
  m_graphUriEdit = new QLineEdit(m_serverWidget);
  m_graphUriEdit->setText("http://localhost:8890/Nettuno");

  serverLayout->addWidget(m_graphUriLabel);
  serverLayout->addWidget(m_graphUriEdit);

  sourceLayout->addWidget(m_pathLabel);
  sourceLayout->addWidget(m_localWidget);
  sourceLayout->addWidget(m_urlLabel);
  sourceLayout->addWidget(m_serverWidget);

  mainLayout->addWidget(sourceGroup);

  // Additional parameters
  auto *paramsGroup = new QGroupBox("Project Parameters", this);
  auto *paramsLayout = new QFormLayout(paramsGroup);


  m_ontologyNamespaceEdit = new QLineEdit(paramsGroup);
  m_ontologyNamespaceEdit->setText("http://www.semanticweb.org/mcodi/ontologies/2024/3/Urban_Ontology");
  paramsLayout->addRow("Ontology Namespace:", m_ontologyNamespaceEdit);

  // auto *ontologyLayout = new QHBoxLayout(paramsGroup);
  m_graphNamespaceEdit = new QLineEdit(paramsGroup);
  m_graphNamespaceEdit->setText("http://www.semanticweb.org/mcodi/ontologies/2024/3/Urban_Ontolog/YTU3D");
  // m_ontologyBrowseButton = new QPushButton("Browse...", paramsGroup);
  // ontologyLayout->addWidget(m_graphNamespaceEdit);
  // ontologyLayout->addWidget(m_ontologyBrowseButton);
  paramsLayout->addRow("Graph Namespace :", m_graphNamespaceEdit);

  mainLayout->addWidget(paramsGroup);

  // Dialog buttons
  auto *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
  mainLayout->addWidget(buttonBox);

  // Initially show local options
  onLocalToggled(true);

  // Connect signals
  connect(m_localRadio, &QRadioButton::toggled, this, &CreateProjectDialog::onLocalToggled);
  connect(m_browseButton, &QPushButton::clicked, this, &CreateProjectDialog::onBrowseClicked);
  // connect(m_ontologyBrowseButton, &QPushButton::clicked, this, [this]() {
  //   QString fileName = QFileDialog::getOpenFileName(
  //           this,
  //           "Select Ontology File",
  //           QString(),
  //           "Ontology Files (*.owl *.rdf *.ttl);;All Files (*.*)");
  //   if (!fileName.isEmpty())
  //     m_graphNamespaceEdit->setText(fileName);
  // });
  connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
  connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
}

void CreateProjectDialog::onLocalToggled(bool checked) {
  m_pathLabel->setVisible(checked);
  m_localWidget->setVisible(checked);
  m_urlLabel->setVisible(!checked);
  m_serverWidget->setVisible(!checked);
}

void CreateProjectDialog::onBrowseClicked() {
  QString fileName = QFileDialog::getOpenFileName(
          this,
          "Select Original Ontology File", // populated ontology file
          QString(),
          "Ontology files (*.rdf *.ttl *.xml *.nt);;All Files (*.*)");

  if (!fileName.isEmpty())
    m_originalPathEdit->setText(fileName);
}

QString CreateProjectDialog::getProjectName() const {
  return m_projectNameEdit->text();
}

bool CreateProjectDialog::isLocal() const {
  return m_localRadio->isChecked();
}

QString CreateProjectDialog::getOriginalPath() const {
  return m_originalPathEdit->text();
}

QString CreateProjectDialog::getServerUrl() const {
  return m_serverUrlEdit->text();
}

QString CreateProjectDialog::getGraphUri() const {
  return m_graphUriEdit->text();
}

QString CreateProjectDialog::getOntologyNamespace() const {
  return m_ontologyNamespaceEdit->text();
}

QString CreateProjectDialog::getGraphNamespace() const {
  return m_graphNamespaceEdit->text();
}

#include "moc_create_project_dialog.cpp"
