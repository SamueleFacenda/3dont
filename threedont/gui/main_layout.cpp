#include "main_layout.h"
#include "dialogs/properties_mapping_selection.h"
#include "dialogs/create_project_dialog.h"
#include <QAction>
#include <QDebug>
#include <QDockWidget>
#include <QLineEdit>
#include <QMenu>
#include <QMessageBox>

MainLayout::MainLayout(ControllerWrapper *controllerWrapper, QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainLayout), controllerWrapper(controllerWrapper) {
  ui->setupUi(this);
  ui->errorLabel->setVisible(false);
  ui->statusbar->showMessage(tr("Loading..."));

  viewer = new Viewer(this);
  viewer->setFocusPolicy(Qt::StrongFocus);
  setCentralWidget(viewer);
  ui->statusbar->showMessage(tr("Ready"), 5000);

  connect(qApp, &QCoreApplication::aboutToQuit, this, &MainLayout::cleanupOnExit);
  connect(viewer, &Viewer::singlePointSelected, this, &MainLayout::singlePointSelected);
}

MainLayout::~MainLayout() {
  qDebug() << "Destroying main layout";
  delete ui;
}

void MainLayout::closeEvent(QCloseEvent *event) {
  qDebug() << "Closing main layout";
  controllerWrapper->stop();
  event->accept();
}

void MainLayout::cleanupOnExit() {
  qDebug() << "Scheduling cleaning up main layout";
  this->deleteLater();
}

void MainLayout::singlePointSelected(unsigned int index) {
  controllerWrapper->viewPointDetails(index);
}

void MainLayout::setStatusbarContent(const QString &content, int seconds) {
  ui->statusbar->showMessage(content, seconds * 1000);
}

void MainLayout::on_executeQueryButton_clicked() {
  QString queryType = ui->queryType->currentText();
  QString query = ui->queryTextBox->toPlainText();
  ui->errorLabel->setVisible(false);

  if (query.isEmpty()) return;

  if (queryType == "select") {
    controllerWrapper->selectQuery(query.toStdString());
  } else if (queryType == "scalar") {
    controllerWrapper->scalarQuery(query.toStdString());
  } else if (queryType == "natural language") {
    controllerWrapper->naturalLanguageQuery(query.toStdString());
  } else if (queryType == "tabular") {
    controllerWrapper->tabularQuery(query.toStdString());
  } else {
    ui->errorLabel->setText("Unknown query type");
    ui->errorLabel->setVisible(true);
  }
}

void MainLayout::on_actionLegend_toggled(bool checked) {
  showLegend = checked;
  if (!checked && legendDock) {
    legendDock->close();
    legendDock = nullptr;
  }
}

void MainLayout::displayNodeDetails(const QStringList &details, const QString &parentId) {
  qDebug() << "Displaying node details for " << parentId;

  if (!isDetailsOpen) {
    QDockWidget *dock = new QDockWidget(tr("Point details"), this);
    dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    dock->setFeatures(QDockWidget::DockWidgetClosable);
    connect(dock, &QDockWidget::visibilityChanged, this, &MainLayout::detailsClosed);

    graphTreeModel = new GraphTreeModel(controllerWrapper, this);
    QTreeView *treeView = new QTreeView(dock);
    treeView->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(treeView, &QTreeView::customContextMenuRequested, this, &MainLayout::onTreeViewContexMenuRequested);
    connect(treeView, &QTreeView::expanded, graphTreeModel, &GraphTreeModel::onRowExpanded);
    connect(treeView, &QTreeView::collapsed, graphTreeModel, &GraphTreeModel::onRowCollapsed);
    treeView->setModel(graphTreeModel);
    dock->setWidget(treeView);
    addDockWidget(Qt::LeftDockWidgetArea, dock);
    isDetailsOpen = true;
  }

  graphTreeModel->onChildrenLoaded(parentId, details);
}

void MainLayout::plotTabular(const QStringList &header, const QStringList &rows) {
  auto dock = new QDockWidget(tr("Tabular data"), this);
  dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
  dock->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable | QDockWidget::DockWidgetClosable);

  int nVars = header.size();
  QTableWidget *tableWidget = new QTableWidget(dock);
  tableWidget->setColumnCount(nVars);
  tableWidget->setRowCount(rows.size() / nVars);
  tableWidget->setHorizontalHeaderLabels(header);
  tableWidget->verticalHeader()->setVisible(false);
  tableWidget->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);

  std::vector<bool> isColorColumn(nVars, false);
  for (int j = 0; j < nVars; ++j) {
    if (header[j].contains("olor")) // heuristic to detect color columns
      isColorColumn[j] = true;
  }

  for (int i = 0; i < rows.size() / nVars; ++i) {
    for (int j = 0; j < nVars; ++j) {
        QTableWidgetItem *item = new QTableWidgetItem();
      if (isColorColumn[j]) {
        QColor color(rows[i * nVars + j]);
        item->setBackground(color);
      }
      QString content = rows[i * nVars + j];
      if (content.contains('#'))
        // split by #, remove prefix of URI
        content = content.split('#')[1];

      if (content.endsWith("'"))
        content = content.left(content.length() - 1);

      item->setText(content);
      tableWidget->setItem(i, j, item);
    }
  }

  dock->setWidget(tableWidget);
  addDockWidget(Qt::LeftDockWidgetArea, dock);
}

void MainLayout::setQueryError(const QString &error) {
  ui->errorLabel->setText(error);
  ui->errorLabel->setVisible(true);
}

void MainLayout::setLegend(const QStringList &colors, const QStringList &labels) {
  if (!showLegend) return;
  if (legendDock) legendDock->close();

  QDockWidget *dock = new QDockWidget(tr("Legend"), this);
  dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea | Qt::BottomDockWidgetArea);
  dock->setFeatures(QDockWidget::DockWidgetClosable);

  QList<QColor> colorList;
  for (const auto &color: colors)
    colorList.append(QColor(color));

  auto *legend = new ColorScaleLegend(colorList, labels, dock);
  connect(legend, &ColorScaleLegend::rangeUpdated, [this](double min, double max) { controllerWrapper->setColorScale(min, max); });

  dock->setWidget(legend);
  addDockWidget(Qt::BottomDockWidgetArea, dock);
  legendDock = dock;
}

void MainLayout::onTreeViewContexMenuRequested(const QPoint &pos) {
  QTreeView *treeView = qobject_cast<QTreeView *>(sender());
  if (!treeView) return;

  QModelIndex index = treeView->indexAt(pos);
  if (!index.isValid()) return;

  QStringList predicatePath = graphTreeModel->getPredicatePath(index);
  std::vector <std::string> predicatePathStd;
  for (const auto &p: predicatePath)
    predicatePathStd.push_back(p.toStdString());

  QString object = graphTreeModel->getObject(index);

  QMenu contextMenu;
  QAction *plotAction = contextMenu.addAction("Plot predicate");
  connect(plotAction, &QAction::triggered, [this, predicatePathStd]() {
    controllerWrapper->scalarWithPredicate(predicatePathStd);
  });

  QAction *annotate = contextMenu.addAction("Annotate");
  connect(annotate, &QAction::triggered, [this, object]() {
    QString subject = object;
    bool ok;
    QString predicate = QInputDialog::getText(this, tr("Annotate"), tr("Predicate Name:"), QLineEdit::Normal,
                                              "type here", &ok);
    if (!ok || predicate.isEmpty()) return;
    QString newObject = QInputDialog::getText(this, tr("Annotate"), tr("Object name or Value:"), QLineEdit::Normal,
                                              "type here", &ok);
    if (!ok || newObject.isEmpty()) return;
    QString authorName = QInputDialog::getText(this, tr("Annotate"), tr("Annotation Author (Name Surname):"), QLineEdit::Normal,
                                               "type here", &ok);
    if (!ok || authorName.isEmpty()) return;

    controllerWrapper->annotateNode(subject.toStdString(), predicate.toStdString(), newObject.toStdString(), authorName.toStdString());
  });

  QAction *selectAll = contextMenu.addAction("Select all");
  connect(selectAll, &QAction::triggered, [this, predicatePathStd, object]() {
    controllerWrapper->selectAllSubjects(predicatePathStd, object.toStdString());
  });

  if (!index.parent().isValid()) {
    QAction *removeAction = contextMenu.addAction("Remove item");
    connect(removeAction, &QAction::triggered, [this, index]() {
      graphTreeModel->removeRow(index.row(), index.parent());
    });
  }

  contextMenu.exec(treeView->viewport()->mapToGlobal(pos));
}

void MainLayout::detailsClosed(bool visible) {
  if (visible) return;
  qDebug() << "Details closed";
  isDetailsOpen = false;
}


void MainLayout::on_actionConfigure_AWS_Connection_triggered() {
  bool ok;
  QString accessKey = QInputDialog::getText(this, tr("Configure AWS connection"), tr("Access Key:"),
                                            QLineEdit::Normal, "type here", &ok);
  if (!ok || accessKey.isEmpty()) return;

  QString secretAccessKey = QInputDialog::getText(this, tr("Configure AWS connection"), tr("Secret Access Key:"),
                                                  QLineEdit::Normal, "type here", &ok);
  if (!ok || secretAccessKey.isEmpty()) return;

  QString region = QInputDialog::getText(this, tr("Configure AWS connection"), tr("Region:"),
                                         QLineEdit::Normal, "eu-west-1", &ok);
  if (!ok || region.isEmpty()) return;

  QString profileName = QInputDialog::getText(this, tr("Configure AWS connection"), tr("Profile Name:"),
                                              QLineEdit::Normal, "type here", &ok);
  if (!ok || profileName.isEmpty()) return;

  controllerWrapper->configureAWSConnection(accessKey.toStdString(), secretAccessKey.toStdString(), region.toStdString(), profileName.toStdString());
}

void MainLayout::on_actionSet_Arguments_PROVISIONAL_triggered() {
  bool ok;
  QString graphUri = QInputDialog::getText(this, tr("Set Arguments"), tr("Graph Uri:"),
                                           QLineEdit::Normal, "type here", &ok);
  if (!ok || graphUri.isEmpty()) return;

  QString ontPath = QInputDialog::getText(this, tr("Set Arguments"), tr("Domain Ontology Path:"),
                                          QLineEdit::Normal, "type here", &ok);
  if (!ok || ontPath.isEmpty()) return;

  QString popOntPath = QInputDialog::getText(this, tr("Set Arguments"), tr("RDF 3DGraph Path:"),
                                             QLineEdit::Normal, "type here", &ok);
  if (!ok || popOntPath.isEmpty()) return;

  QString ontNamespace = QInputDialog::getText(this, tr("Set Arguments"), tr("Domain Ontology Namespace:"),
                                               QLineEdit::Normal, "type here", &ok);
  if (!ok || ontNamespace.isEmpty()) return;

  QString populatedNamespace = QInputDialog::getText(this, tr("Set Arguments"), tr("3DGraph-specific Namespace:"),
                                                     QLineEdit::Normal, "type here", &ok);
  if (!ok || populatedNamespace.isEmpty()) return;

  QString virtuosoIsql = QInputDialog::getText(this, tr("Set Arguments"), tr("Virtuoso ISQL Path:"),
                                               QLineEdit::Normal, "type here", &ok);
  if (!ok || virtuosoIsql.isEmpty()) return;

  controllerWrapper->provisionalSetArgs(graphUri.toStdString(), ontPath.toStdString(), popOntPath.toStdString(), ontNamespace.toStdString(), populatedNamespace.toStdString(), virtuosoIsql.toStdString());
}

void MainLayout::on_actionAdd_Sensor_triggered() {
  bool ok;
  QString sensorName = QInputDialog::getText(this, tr("Add a Sensor"), tr("Desired Name of the Sensor:"),
                                             QLineEdit::Normal, "type here", &ok);
  if (!ok || sensorName.isEmpty()) return;

  QString objectName = QInputDialog::getText(this, tr("Add a Sensor"), tr("Name of the Described Entity:"),
                                             QLineEdit::Normal, "type here", &ok);
  if (!ok || objectName.isEmpty()) return;

  QString propertyName = QInputDialog::getText(this, tr("Add a Sensor"), tr("Name of the Measured Property:"),
                                               QLineEdit::Normal, "type here", &ok);
  if (!ok || propertyName.isEmpty()) return;

  QString certPemPath = QInputDialog::getText(this, tr("Add a Sensor"), tr("Path for the Certificate:"),
                                              QLineEdit::Normal, "type here", &ok);
  if (!ok || certPemPath.isEmpty()) return;

  QString privateKeyPath = QInputDialog::getText(this, tr("Add a Sensor"), tr("Path for the Private Key:"),
                                                 QLineEdit::Normal, "type here", &ok);
  if (!ok || privateKeyPath.isEmpty()) return;

  QString rootCaPath = QInputDialog::getText(this, tr("Add a Sensor"), tr("Path for the Root CA:"),
                                             QLineEdit::Normal, "type here", &ok);
  if (!ok || rootCaPath.isEmpty()) return;

  QString mqttTopic = QInputDialog::getText(this, tr("Add a Sensor"), tr("MQTT Topic:"),
                                            QLineEdit::Normal, "type here", &ok);
  if (!ok || mqttTopic.isEmpty()) return;

  QString clientId = QInputDialog::getText(this, tr("Add a Sensor"), tr("Client ID:"),
                                           QLineEdit::Normal, "type here", &ok);
  if (!ok || clientId.isEmpty()) return;


  controllerWrapper->addSensor(sensorName.toStdString(), objectName.toStdString(), propertyName.toStdString(), certPemPath.toStdString(), privateKeyPath.toStdString(), rootCaPath.toStdString(), mqttTopic.toStdString(), clientId.toStdString());
}

void MainLayout::on_actionUpdate_Sensors_and_Reason_triggered() {
  controllerWrapper->updateSensorsAndReason();
}

void MainLayout::setProjectList(const QStringList &projects) {
  auto *fileMenu = ui->menubar->findChild<QMenu *>("menuFile");
  auto *openProjectMenu = fileMenu->findChild<QMenu *>("menuOpen_project");
  if (!openProjectMenu) {
    openProjectMenu = new QMenu(tr("Open project"), fileMenu);
    openProjectMenu->setObjectName("menuOpen_project");
    fileMenu->addMenu(openProjectMenu);
  } else {
    openProjectMenu->clear();
  }

  for (const QString &project: projects) {
    QAction *action = openProjectMenu->addAction(project);
    connect(action, &QAction::triggered, this, [this, project]() {
      controllerWrapper->openProject(project.toStdString());
    });
  }
}

QByteArray MainLayout::sendViewerCommand(const QByteArrayView &message) {
  const char* data = message.constData();
  qint64 size = message.size();
  return viewer->processCommand(data, size);
}

void MainLayout::on_actionCreate_project_triggered() {
  CreateProjectDialog dialog(this);
  if (dialog.exec() != QDialog::Accepted) {
    return;
  }
  
  QString projectName = dialog.getProjectName();
  if (projectName.isEmpty()) {
    QMessageBox::warning(this, tr("Create Project"), tr("Project name cannot be empty."));
    return;
  }
  
  QString graphUri = dialog.getGraphUri();
  QString ontologyNamespace = dialog.getOntologyNamespace();
  QString graphNamespace = dialog.getGraphNamespace();
  QString ontologyPath = dialog.getOntologyNamespace();
  
  if (graphUri.isEmpty() || ontologyNamespace.isEmpty() || ontologyPath.isEmpty()) {
    QMessageBox::warning(this, tr("Create Project"), tr("All parameters are required."));
    return;
  }
  
  bool isLocal = dialog.isLocal();
  QString originalPath;
  QString dbUrl;
  
  if (isLocal) {
    originalPath = dialog.getOriginalPath();
    if (originalPath.isEmpty()) {
      QMessageBox::warning(this, tr("Create Project"), tr("Please select an original file."));
      return;
    }
    dbUrl = ""; // Empty for local projects
  } else {
    dbUrl = dialog.getServerUrl();
    if (dbUrl.isEmpty()) {
      QMessageBox::warning(this, tr("Create Project"), tr("Please enter a server URL."));
      return;
    }
    originalPath = ""; // Empty for server projects
  }
  
  controllerWrapper->createProject(projectName.toStdString(), dbUrl.toStdString(), graphUri.toStdString(), 
                                   graphNamespace.toStdString(), isLocal, originalPath.toStdString(),
                                   ontologyNamespace.toStdString());
}

QStringList MainLayout::getPropertiesMapping(const QStringList &properties, const QStringList &words, const QStringList &defaults) {
  return PropertiesMappingDialog::getPropertiesMapping(this, properties, words, defaults);
}

void MainLayout::on_actionRotate_camera_around_triggered() {
  controllerWrapper->rotateAround();
}

void MainLayout::on_actionReset_query_results_buffer_triggered() {
  controllerWrapper->resetQueryResultBuffer();
}

#include "moc_main_layout.cpp"
