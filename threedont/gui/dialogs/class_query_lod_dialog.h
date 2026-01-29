#ifndef CLASS_QUERY_LOD_DIALOG_H
#define CLASS_QUERY_LOD_DIALOG_H

#include <QDialog>
#include <QList>

class QLabel;
class QSlider;
class QDialogButtonBox;

class ClassQueryLodDialog : public QDialog {
    Q_OBJECT

public:
    explicit ClassQueryLodDialog(QWidget *parent = nullptr);
    ~ClassQueryLodDialog() override;

    int lodValue() const;

private:
    QList<int> lodValues;
    QLabel* label;
    QSlider* slider;
    QDialogButtonBox* buttonBox;

private slots:
    void on_slider_valueChanged(int value);
};

#endif // CLASS_QUERY_LOD_DIALOG_H
