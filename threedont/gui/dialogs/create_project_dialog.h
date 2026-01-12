#ifndef THREEDONT_CREATE_PROJECT_DIALOG_H
#define THREEDONT_CREATE_PROJECT_DIALOG_H

#include <QDialog>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QRadioButton>
#include <QVBoxLayout>

class CreateProjectDialog : public QDialog {
  Q_OBJECT

public:
  explicit CreateProjectDialog(QWidget *parent = nullptr);

  QString getProjectName() const;
  bool isLocal() const;
  QString getOriginalPath() const;
  QString getServerUrl() const;
  QString getGraphUri() const;
  QString getOntologyNamespace() const;
  QString getGraphNamespace() const;

private slots:
  void onLocalToggled(bool checked);
  void onBrowseClicked();

private:
  void setupUI();

  QLineEdit *m_projectNameEdit;
  QRadioButton *m_localRadio;
  QRadioButton *m_serverRadio;
  QLineEdit *m_originalPathEdit;
  QPushButton *m_browseButton;
  QLineEdit *m_serverUrlEdit;
  QLineEdit *m_graphUriEdit;
  QLineEdit *m_ontologyNamespaceEdit;
  QLineEdit *m_graphNamespaceEdit;
  QPushButton *m_ontologyBrowseButton;
  QLabel *m_pathLabel;
  QLabel *m_urlLabel;
  QLabel *m_graphUriLabel;
  QWidget *m_localWidget;
  QWidget *m_serverWidget;
};

#endif // THREEDONT_CREATE_PROJECT_DIALOG_H
